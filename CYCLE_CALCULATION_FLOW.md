# Prosperity 사이클 계산 완전 플로우 (팩트만)

## 1. 전체 사이클 계산 구조

```
Workload (신경망) 
    ↓
각 레이어별 처리:
  - FC / Conv2D
  - LIF Neuron  
  - Attention
  - LayerNorm
    ↓
각 레이어의 total_cycles 계산
    ↓
전체 합산 (일부 겹침 고려)
    ↓
최종 total_cycles
    ↓
에너지 계산 (총 사이클 × 전력)
```

---

## 2. FC/SpMM 사이클 계산 (메인 연산)

### **2.1 입력 데이터**

```python
# simulator.py: run_fc_cuda()
operator.activation_tensor.sparse_map  # 실제 활성화 (비트맵)
operator.weight_tensor               # 가중치
M, K, N = input_shape[0], input_shape[1], operator.weight_tensor.shape[1]
# M = 배치×시간스텝×시퀀스
# K = 입력 채널
# N = 출력 채널
```

### **2.2 타일링 (타일 단위 처리)**

```python
tile_size_M = 256       # 가능한 크기 (DSE로 조정 가능)
tile_size_K = 16        # 가능한 크기 (DSE로 조정 가능)
tile_size_N = 128       # PE 배열 크기

tile_num_M = ceil(M / tile_size_M)
tile_num_K = ceil(K / tile_size_K)
tile_num_N = ceil(N / tile_size_N)
```

### **2.3 Product Sparsity 추출**

```python
# CUDA 커널 호출 (prosparsity_cuda.cu)
prosparsity_act, prefix_array = prosparsity_engine.find_product_sparsity(
    input_act, tile_size_M, tile_size_K
)

# 결과:
# - prosparsity_act: M×K 행렬, Product Sparsity 적용 후
# - prefix_array: 각 행의 prefix row 인덱스 (-1 = 없음)
```

### **2.4 핵심: 사이클 계산 (3가지 방식)**

#### **방식 A: issue_type=2 (기본값)**

```python
# simulator.py: run_fc_cuda() 라인 207-209
compute_cycles = (torch.sum(prosparsity_act != 0).item() +    # NNZ 개수
                  torch.sum(all_zero_row).item() -            # 행 스킵 이득
                  torch.sum(all_zero_row_ori).item()) \
                * tile_num_N                                  # 타일 반복

# 예시:
# NNZ = 1000개
# 행 스킵 (새로 생겨난) = 50개
# 행 스킵 (원래) = 100개
# tile_num_N = 4 (128 PE, N=512 → 512/128=4)
#
# compute_cycles = (1000 + 50 - 100) * 4 = 3800 사이클
```

**가정:**
- 비영-제로 1개 = PE에서 1 사이클 처리
- Product Sparsity로 새로 생겨난 영-제로 행 = 1 사이클 (검사 오버헤드)
- 원래부터 영-제로인 행 = 이미 제외됨

#### **방식 B: issue_type=1 (Prefix Tree 깊이 고려)**

```python
# simulator.py: run_fc_cuda() 라인 214-221
if self.accelerator.issue_type == 1:
    compute_cycles = 0
    for i in range(prefix_array.shape[0]):  # 타일마다
        for j in range(prefix_array.shape[1]):
            cur_forest = construct_prosparsity_forest(cur_prefix)
            depth = nx.dag_longest_path_length(cur_forest)
            cur_issue_cycles = depth // 4 * tile_size_M * tile_num_N
            compute_cycles += max(cur_issue_cycles, cur_compute_cycles)

# 예시:
# prefix tree 깊이 = 8
# issue_cycles = 8 / 4 * 256 * 4 = 2048
# 
# vs 직접 계산 = 3800
#
# 최종 = max(2048, 3800) = 3800
```

**추가 가정:**
- Prefix tree 깊이 d = d/4 사이클 추가 지연
- 깊이 4 = 1 사이클 오버헤드

### **2.5 전처리 사이클 (Product Sparsity 생성)**

```python
# simulator.py: run_fc() 라인 320
preprocess_cycles = (get_prosparsity_cycles(input_act) + 
                     M // self.accelerator.num_popcnt) * tile_num_N

# get_prosparsity_cycles: 
# = 각 행의 NNZ > 1인 행 개수
#   (영-제로가 아니고, 스칼라가 아닌 행만 처리 필요)

# M // num_popcnt:
# = M행을 num_popcnt(=8) 병렬 처리 → M/8 사이클

# 예시:
# 처리 필요 행 = 800개
# POPCOUNT 병렬도 = 8
# tile_num_N = 4
# preprocess_cycles = (800 + 256/8) * 4 = 3264 사이클
```

**가정:**
- 각 행 검사 = 1 사이클
- POPCOUNT = 8개 병렬 (num_popcnt=8)
- 전처리와 계산 오버랩 가능 (나중에 max() 취함)

### **2.6 최종 연산 사이클**

```python
# simulator.py: run_fc() 라인 551-552
stats.compute_cycles = max(compute_cycles, preprocess_cycles)
stats.preprocess_stall_cycles = max(0, preprocess_cycles - compute_cycles)
```

**해석:**
- 계산과 전처리 중 긴 것 = 진짜 걸린 시간
- 짧은 것 = 오버랩됨

### **2.7 메모리 지연**

```python
# simulator.py: run_fc() 라인 546-550
init_mem_access = min(tile_size_K, K) * min(tile_size_N, N) * nbits  # 첫 타일
total_mem_access = stats.reads['dram'] + stats.writes['dram']
middle_mem_access = total_mem_access - init_mem_access

init_latency = init_mem_access // self.accelerator.mem_if_width
middle_latency = middle_mem_access // self.accelerator.mem_if_width

mem_stall_cycles = init_latency + max(0, middle_latency - compute_cycles)
```

**계산:**
- init_latency = 첫 타일 로드 지연 (병렬화 불가)
- middle_latency = 나머지 메모리 지연
- 계산이 빠르면 메모리 대기 발생 (mem_stall_cycles > 0)
- 계산이 느리면 메모리 오버랩됨 (mem_stall_cycles = 0)

**예시:**
```
mem_if_width = 1024 비트

첫 타일: 256 * 128 * 8 = 262,144 비트 → 256 사이클
나머지: 10,000,000 비트 → 9,766 사이클

compute_cycles = 3800

init_latency = 256
middle_latency = 9,766
mem_stall_cycles = 256 + max(0, 9,766 - 3800) = 256 + 5,966 = 6,222

total_cycles = compute_cycles + mem_stall_cycles
            = 3800 + 6,222 = 10,022
```

---

## 3. LIF 뉴런 사이클

```python
# simulator.py: run_LIF() 라인 489-495
num_round = ceil(num_neuron * batch_size / LIF_array_size)
compute_cycles = num_round * time_steps * 2

# 가정:
# - LIF 배열 크기 = 32개
# - 뉴론 수 = 256개
# - 타임스텝 = 4
#
# num_round = ceil(256 / 32) = 8
# compute_cycles = 8 * 4 * 2 = 64 사이클
```

**가정:**
- 각 라운드 × 타임스텝 = 2 사이클
- 1 = 덧셈, 1 = 곱셈 (시각적 레이턴시)

---

## 4. Attention 사이클

```python
# simulator.py: run_attention() 라인 518-674
# 세 가지 타입 지원:

# A. spikformer / spikebert:
# K×V 계산 (FC로 분해)
cur_stats_kv = self.run_fc(eq_fc_1, ...)  # 재귀 호출
stats.total_cycles += cur_stats_kv.total_cycles

# Q×KV 계산 (FC로 분해)  
cur_stats_qkv = self.run_fc(eq_fc_2, ...)
stats.total_cycles += cur_stats_qkv.total_cycles

# B. sdt (Spiking Transformer):
# Q·K 계산 (내적)
compute_cycles += num_op // adder_array_size
# Softmax (SFU, 나눗셈, 곱셈)
# V 곱셈 (내적)
compute_cycles += num_op // adder_array_size

# C. spikingbert:
# Q·K 계산
cur_stats_qk = self.run_fc(eq_fc_qk, True, True)
qk_cycles = cur_stats_qk.total_cycles

# Softmax
SFU_cycles = seq_len * seq_len // num_exp       # 지수
adder_cycles = seq_len * seq_len // adder_size  # 합계
div_cycles = seq_len // num_div                 # 역수
mult_cycles = seq_len * seq_len // mult_size    # 정규화
softmax_cycles = SFU_cycles + mult_cycles

# S·V 계산
compute_cycles += max(qk_cycles + sv_cycles + adder_cycles, softmax_cycles)
```

**계산 특징:**
- Attention = 여러 FC의 합 (재귀 호출)
- Softmax = 특수 함수 유닛 (SFU) 사용

---

## 5. 전체 통합

```python
# simulator.py: run_simulation() 라인 130-170
total_stats = Stats()
for key, value in stats.items():  # 각 레이어
    total_stats += value
    
    # LIF는 이전 연산과 겹칠 수 있음
    if key.startswith('lif') and (이전이 fc/conv/layernorm):
        # LIF 지연만 추가 (계산이 더 빠르면)
        total_stats.total_cycles += value.LIF_latency
    else:
        # 전체 사이클 추가
        total_stats.total_cycles += value.total_cycles

# 최종:
print("total cycles: ", total_stats.total_cycles)
```

**겹침 로직:**
- LIF는 계산 집약적이지 않음 → 이전 FC/Conv와 겹칠 수 있음
- LIF_latency = 더 짧은 시간 (병렬화된 처리)
- 파이프라인 깊이: LIF 고려 (32 PE, 짧은 작업)

---

## 6. 에너지 계산 (Power/Area는 별도)

```python
# energy.py: get_total_energy()

# 온칩 파워 (Synopsys로 측정한 값 사용)
on_chip_power_dict = {
    'Prosperity': 446.5,    # mW  ← Synopsys Design Compiler 결과
    'Eyeriss': 1410.5,
    'SATO': 319.5,
    ...
}

processing_time = total_cycles / (500 * 1000 * 1000)  # 초 단위 (500MHz)
on_chip_energy = on_chip_power * processing_time      # mJ

# DRAM 에너지
dram_access = read_position(mem_reference.csv, model, "mem_access")
dram_energy = 12.45 * dram_access * 1e-9  # pJ/비트 × 접근 수

total_energy = on_chip_energy + dram_energy
```

**중요:**
- **온칩 파워**: Synopsys Design Compiler로 측정
- **DRAM 에너지**: DRAMsim3 기반 에너지/비트
- **mem_reference.csv**: Prosperity 시뮬레이션에서 계산한 DRAM 접근 수

```csv
# reference/mem_reference.csv 예시
,spikformer_cifar10,spikformer_cifar100,...
mem_access,120000000,150000000,...
```

---

## 7. Baseline 비교 (동일 방식)

```python
# 각 baseline도 동일 workload에 대해 사이클 계산

# SATO:
compute_cycles = max(PE_load) * output_dim

# Eyeriss (14×12 Systolic):
compute_cycles = ceil(output_dim / 12) * (input_dim / 14)

# PTB:
compute_cycles += input_length * repeat_times * time_window
mem_stall = init_latency + max(0, middle_latency - compute_cycles)

# 모두 같은 입력 데이터 사용:
# - data/*.pkl (실제 신경망 실행 데이터)
# - 동일한 타임스텝, 배치, 모델 구조
```

---

## 8. 출력

```
Output Example:
total cycles: 10,022,156
time: 20.044312 seconds (at 500MHz)
total ops: 256,000,000
mem access: 125,000,000 bits
```

**변환:**
```
사이클 → 시간: cycles / 500e6 = 시간(초)
시간 → 에너지: 시간 × 전력 = 에너지
```

---

## 9. Synopsys Design Compiler 사용 위치

**명시된 위치:**
- README.md: "Power and area stats evaluated by synopsys design compiler"

**실제 사용:**
- Power 값 추출 (on_chip_power_dict)
- Area 값 (논문 Figure 10)

**코드에서의 사용:**
```python
# energy.py의 상수값들이 SDC 결과
on_chip_power_dict = {
    'Prosperity': 446.5,  # ← SDC 결과: 온칩 파워
    ...
}

# 이 값들은 논문의 실제 측정값
# (코드에는 하드코딩되어 있고, 도출 과정은 비공개)
```

**SDC 미포함:**
- 사이클 계산 (Python 모델)
- 메모리 접근 계산 (Python 모델)
- 버퍼 영역 (CACTI 사용)

**SDC 포함:**
- 온칩 전력 (하드코딩된 상수)
- 면적 (논문 Figure 10)

---

## 10. 정리: 사이클 계산 소스

| 컴포넌트 | 계산 방식 | 소스 |
|---------|---------|------|
| **SpMM 연산** | NNZ 개수 × tile 수 | Python 모델 |
| **Prefix Tree 깊이** | 그래프 최장경로 | networkx (Python) |
| **LIF** | round × timestep × 2 | Python 모델 |
| **Attention** | FC 재귀 + SFU | Python 모델 |
| **메모리 지연** | 접근 수 / 1024비트 | Python 모델 |
| **온칩 전력** | 하드코딩 상수 | **Synopsys SDC** |
| **DRAM 에너지** | 접근 수 × 12.45pJ | DRAMsim3 |
| **버퍼 면적** | CACTI 도구 | **CACTI** |

---

## 11. 사이클 모델의 가정 명시

| 가정 | 코드 위치 | 근거 |
|------|---------|------|
| 비영-제로 1개 = 1 사이클 | run_fc() L320 | 암묵적 |
| LIF = 2 사이클 | run_LIF() L495 | 주석: "덧셈+곱셈" |
| 메모리 대역폭 = 1024비트 | accelerator.py | 기본값 (설정 가능) |
| PE 배열 = 128 (N 차원) | accelerator.py | 기본값 (설정 가능) |
| 파이프라인 오버랩 | run_attention() | 재귀 호출 max() 사용 |

