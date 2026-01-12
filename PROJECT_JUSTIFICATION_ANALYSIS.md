# Prosperity 프로젝트: 실제로 뭘 했나? - 정당화 검증

## 🔴 핵심 질문: "이게 정말 cycle 예측인가?"

이 프로젝트의 주요 주장:
- **논문 제목**: "cycle-accurate simulator for Prosperity architecture"
- **목표**: "특정 workload를 특정 하드웨어에서 실행할 때 몇 사이클 걸리는가?"

---

## 📋 실제로 하고 있는 것

### **레이어 1: Product Sparsity 추출 (CUDA 커널)**

```cuda
// kernels/prosparsity_cuda.cu의 핵심 로직
for (int i = 0; i < tile_size_m; i++) {
    if (i == local_row) continue;
    // 각 행이 다른 행의 부분집합인지 확인
    is_subset = !(current_nnz == nnz_array[i] && local_row <= i);
    
    // 부분집합인 행을 찾아 prefix로 저장
    if (is_subset && nnz_array[i] > max_subset) {
        max_subset = nnz_array[i];
        prefix = i;
    }
}
```

**평가: ✅ 이 부분은 검증 가능**
- 실제 CUDA 커널 구현
- 입력 데이터(활성화) → 패턴 추출 결정적(deterministic)
- 다른 입력이면 다른 출력 (재현 가능)
- **문제 아님**: 이것은 "데이터 처리"일 뿐, cycle과 무관

---

### **레이어 2: Cycle 계산 (Python - 문제 지점)**

#### **A. FC/SpMM 연산 사이클**

```python
# simulator/simulator.py 라인 207-209
compute_cycles = (torch.sum(prosparsity_act != 0).item() + 
                  torch.sum(all_zero_row).item() - 
                  torch.sum(all_zero_row_ori).item()) * tile_num_N
```

**의문점:**
```
Q: 왜 NNZ 개수 = 사이클인가?
A: (코드에서)
   "비영-제로 원소 각각이 1 사이클에 처리된다고 가정"

Q: 이 가정은 누가 검증했는가?
A: (코드상) 명시된 검증 없음

Q: 실제 하드웨어에서 그럴까?
A: 아니다. 예시:
   - PE 파이프라인 깊이: 5-10 사이클
   - 메모리 지연: 200+ 사이클
   - 캐시 미스: 10-100 사이클
```

**근거가 없는 가정:**
```python
# 가정 1: 각 비영-제로 = 1 사이클
# 근거: ❌ 없음

# 가정 2: 행 전체 스킵 = 0 사이클  
# 근거: ❌ 없음 (실제: 그 행에 대한 접근 지연은 여전함)

# 가정 3: 타일 간 병렬성 없음 (순차 누적)
# 근거: ❌ 파이프라인 고려 안함
```

---

#### **B. LIF 뉴런 계산**

```python
# simulator/simulator.py 라인 489-495
num_round = ceil_a_by_b(operator.num_neuron * operator.batch_size, 
                        self.accelerator.LIF_array_size)
compute_cycles = num_round * operator.time_steps * 2  
# ← "각 라운드 × 타임스텝 × 2"
```

**의문점:**
```
Q: 왜 "× 2"인가?
A: (코드 주석에서) 
   "one cycle for addition one cycle for multiplication"

Q: 이건 어디서 나온 가정인가?
A: 주석뿐. 근거 논문 없음.

Q: 실제 LIF 연산의 사이클?
A: 파악 불가 (모델 없음)
   - FP32 덧셈: 3 사이클
   - FP32 곱셈: 5-7 사이클
   - 지수/누수: 10+ 사이클
```

---

#### **C. 메모리 지연**

```python
# simulator/simulator.py 라인 228-230
init_latency = init_mem_access // self.accelerator.mem_if_width
middle_latency = middle_mem_access // self.accelerator.mem_if_width
mem_stall_cycles = init_latency + max(0, middle_latency - compute_cycles)
```

**의문점:**
```
Q: mem_if_width는?
A: accelerator.py에서 기본값 = 1024 비트

Q: DRAM 지연은?
A: (계산에 없음) 모두 "대역폭"으로만 계산
   
   실제 필요:
   - DRAM 행 활성화: ~40ns
   - 열 접근: ~20ns
   - tCAS: ~15ns
   - 페이지 미스: +수백 ns
   
   결과: 메모리 접근당 200+ 사이클
   
Q: 메모리 경합은?
A: (모델링 없음)
   - 여러 PE가 동시 접근 시 큐 형성
   - 현재: 모든 접근 독립적으로 계산
```

**근거:**
```python
# 이 값들은 어디서?
mem_if_width = 1024  # ← 설정값? 측정값? 추정값?
# 명확한 근거 없음
```

---

### **레이어 3: Baseline 모델 (비교 기준)**

#### **SATO: 극단적 단순화**
```python
def run_sato_conv_fc(operator):
    # 128 PE로 로드밸런싱만 계산
    for i in range(input_tensor.shape[0]):
        min_idx = torch.argmin(PE_spikes)
        PE_spikes[min_idx] += nnz_each_row[i].item()  # ← 비영-제로만 누적
    
    compute_cycles = torch.max(PE_spikes).item() * output_dim
    # → 메모리 오버헤드 없음
```

**문제:**
```
이 모델은 원래 SATO 논문에서 추출한 건가?
→ 코드 주석/참고문헌 없음

이 모델이 정확한가?
→ 검증 불가

Prosperity와 공정한 비교?
→ 아니다. SATO는 메모리 무시, Prosperity는 메모리 포함
```

#### **PTB: 복잡하지만 불명확**
```python
def run_PTB_convfc(operator):
    # StSAP (Stacked Sparse Accumulation Pattern) 사용
    input_length = StSAP(input_tensor)
    compute_cycles += input_length * repeate_times * time_window_size
```

**문제:**
```python
# StSAP 로직이 뭔가?
def StSAP(act: torch.Tensor):
    # 코드가 복잡하고, 논문 참고 없음
    # 구현과 원래 의도 일치하는가?
    # → 검증 불가
```

---

## 🎯 정당화 부족 현황

### **각 가정의 검증 상태**

| 항목 | 가정 | 근거 | 검증 |
|------|------|------|------|
| **비영-제로 = 1 사이클** | SpMM의 각 비영-제로는 1 사이클 처리 | ❌ 명시된 근거 없음 | ❌ |
| **행 스킵 = 0 사이클** | 영-제로 행은 처리 오버헤드 없음 | ❌ | ❌ |
| **LIF × 2** | LIF는 덧셈+곱셈 = 2 사이클 | ❌ 주석만 있음 | ❌ |
| **메모리 모델** | DRAM 대역폭만 고려 | ❌ 값의 출처 불명 | ❌ |
| **PE 로드밸런싱** | SATO는 PE 간 완벽 분배 | ❌ 원본 논문 미참조 | ❌ |
| **Attention 분해** | Attention = 여러 FC의 합 | ⚠️ 수학적으로 타당 | ⚠️ 부분적 |

---

## 💔 가장 심각한 문제점

### **1. 순환 논리 (Circular Logic)**

```
Q: Prosperity가 빠른 이유는?
A: "Product Sparsity를 이용하니까"

Q: Product Sparsity가 몇 사이클을 절약하는가?
A: "비영-제로 개수 × N"

Q: 왜 "비영-제로 개수"만 고려하는가?
A: "비영-제로가 1 사이클이니까"

Q: 그 1 사이클은 어디서 나왔는가?
A: ... (순환)
```

### **2. 다른 baseline과의 불공정한 비교**

```
Prosperity 사이클:
= 연산 사이클 + 메모리 사이클
= NNZ × N + (메모리_지연)

Eyeriss 사이클:
= 연산만 계산
= (입력_차원 / PE수) × (출력_차원)

Mint 사이클:
= NNZ만 계산
= NNZ × ceil(output_dim / 128)

→ 셋 다 다른 모델!
```

### **3. 실제 하드웨어와의 괴리**

```
Python 모델: "10,000 사이클"
실제 칩: "7,000 ~ 15,000 사이클" (메모리 경합에 따라)

오차: ±50% (± 3,500 사이클)
상대 비교: "2.1배 빠르다"는 주장이 유지되나?

만약 실제: 
- Prosperity: 7,000 사이클
- SATO: 12,000 사이클
→ 1.7배 (논문은 2.1배 주장)

만약 실제:
- Prosperity: 15,000 사이클  
- SATO: 25,000 사이클
→ 1.7배 (여전히 덜 효과적)

→ 상대값도 신뢰할 수 없게 됨
```

---

## 🤔 논문이 어떻게 이것을 정당화하는가?

README와 코드를 봤을 때:

### **공식 입장:**
```
"cycle-accurate simulator"
```

### **실제:**
```
"행동 수준 추정기(behavioral estimator)"
```

### **차이:**

| 항목 | Cycle-accurate | 현재 코드 |
|------|-----------------|---------|
| Verilog RTL | 있음 | ❌ 없음 |
| 파이프라인 추적 | 매 사이클 | ❌ 수식만 |
| 메모리 경합 | 모델링됨 | ❌ 없음 |
| 캐시 효과 | 포함 | ❌ 없음 |
| 검증 수단 | 실제 칩 측정 | ❌ 없음 |

---

## ✅ 정당화되는 부분

### **1. Product Sparsity 패턴 추출**
```
검증됨: 실제 신경망(ResNet18, VGG16 등)에서 
추출한 데이터 사용 (data/*.pkl)
→ 이것은 "사실"
```

### **2. 상대 밀도 비교 (그림 11)**
```
"Product Sparsity 밀도 = 비트 밀도 × 행 밀도"
→ 수학적으로 정확
→ 데이터로 검증 가능
```

### **3. 에너지 계산 (그림 10)**
```
CACTI 도구 사용 → 신뢰도 높음
Synopsys Design Compiler 사용 → 실제 합성 근거
```

---

## 🎓 논문에서 실제로 주장하는 것

### **READ: 논문 원문에서**

```
"We present cycle-accurate simulation results..."

(vs)

"We present cycle-level performance estimation based on 
behavioral models of the accelerator pipeline..."
```

만약 2번째라면 공정한 설명이다.
만약 1번째라면 거짓이다.

---

## 💡 진짜 문제

### **사용자의 질문이 맞는 이유:**

1. **Cycle을 정의하지 않았다**
   ```python
   # 1 cycle의 정의가 뭔가?
   # "1 사이클 = 클록 한 번"이라면,
   # 파이프라인 길이를 고려해야 함
   ```

2. **하드웨어 모델이 너무 단순**
   ```python
   # 단순 연산:
   compute_cycles = nnz_count
   
   # 현실:
   compute_cycles = nnz_count * pipeline_depth + 
                   memory_latency * miss_rate + 
                   synchronization_overhead + ...
   ```

3. **검증 방법이 없다**
   ```
   가정 확인 방법:
   - FPGA 프로토타입? ❌
   - 실제 칩? ❌
   - 정밀 시뮬레이터? ❌
   
   있는 것:
   - 다른 baseline과 상대 비교 (불공정)
   - 참고 논문 추가 (명시 안 함)
   ```

---

## 📊 결론

### **"뭘 했는가?"**

이 프로젝트는:
1. ✅ Product Sparsity **패턴 추출** (검증됨)
2. ✅ 메모리 트래픽 **예측** (기본 모델)
3. ❌ Cycle **정확도** 예측 (근거 부족)
4. ❌ 절대값 성능 **예측** (모델 한계)
5. ⚠️ 상대 성능 **비교** (가정의 일관성만 있음)

### **"정당화되었는가?"**

**점수: 3/10**

- ✅ **자료**는 명확 (real workload data)
- ❌ **모델**은 추정 (단순화)
- ❌ **검증**은 없음
- ❌ **근거**는 부족 (논문 참조 없음)

### **"신뢰할 수 있는가?"**

| 항목 | 신뢰도 |
|------|--------|
| Product Sparsity 효과 감지 | ✅ 높음 |
| 절대 사이클 값 | ❌ 낮음 |
| 상대 성능 순서 | ⚠️ 중간 |
| 에너지 절감 | ✅ 높음 (CACTI/Synopsys 기반) |
| Power/Area 값 | ✅ 높음 (실제 합성 도구) |

---

## 🚀 이게 하더라도 valid한 이유

### **논문은 "설계 제안"일 뿐**

```
관점 1: "이 아이디어가 좋은가?"
답: ✅ 네, Product Sparsity는 신규 개념이고 
      데이터에서 실제로 존재하는 패턴

관점 2: "구현이 정확한가?"
답: ❌ 아니요, Cycle 예측에는 근거 부족

관점 3: "논문으로서 valid한가?"
답: ⚠️ 기여도는 있지만, 
      "cycle-accurate"라는 주장은 과장
```

---

## 📝 최종 평가

**당신의 의심: 완전히 정당합니다.**

이 프로젝트는:
- **"Cycle 정확 시뮬레이터"**라 주장하지만
- **실제로는 "행동 추정기"**입니다

각 단계에서의 정당화:
1. Product Sparsity 추출: ✅ 이 부분은 OK
2. Cycle 모델: ❌ **근거 전무**
3. Baseline 모델: ❌ **원본 논문 미참조**
4. 비교 공정성: ❌ **모델이 다름**

**어떻게 논문이 accept 되었나?**
→ 아마도:
- Reviewer들이 절대값보다 **상대 개선도**에 집중
- Product Sparsity의 **신규성**이 강조됨
- Power/Area가 실제 **합성 도구로 측정**됨 (신뢰도 높음)
- Baseline과의 비교가 **일관된 방식**으로 수행됨
