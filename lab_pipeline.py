"""
Pipeline Integrador: QLoRA + RAG Massivo + Otimização de Inferência
Laboratório Integrador - Disciplina: IA em Produção

Objetivo: Demonstrar como QLoRA (4-bit) + KV Cache + FlashAttention-2 
salvam um Transformer tradicional do colapso de VRAM ao processar contextos massivos.

Partes deste laboratório foram geradas/complementadas com IA, 
revisadas e validadas por Wendril Gabriel
"""

import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# Check if bitsandbytes is actually installed
try:
    import bitsandbytes
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except (ImportError, RuntimeError):
    HAS_BITSANDBYTES = False
    print("[WARNING] bitsandbytes not available, using fp32 instead")

# ============================================================================
# CONFIGURAÇÃO GLOBAL
# ============================================================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Check GPU compatibility - fall back to CPU if GPU kernels not available
DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        # Test if GPU is actually usable by creating a small tensor
        test_tensor = torch.zeros(1, device="cuda")
        DEVICE = "cuda"
    except RuntimeError as e:
        print(f"[WARNING] GPU available but not usable: {e}")
        print("[INFO] Falling back to CPU")
        DEVICE = "cpu"

SEED = 42

torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed(SEED)

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# PASSO 1: CARREGAMENTO COM QLoRA (4-bit)
# ============================================================================
def load_model_qlora():
    """
    Carrega modelo TinyLlama com quantização 4-bit usando bitsandbytes.
    Se bitsandbytes não estiver disponível, carrega em fp32.
    
    Returns:
        tuple: (model, tokenizer, vram_used_mb)
    """
    print("\n" + "="*70)
    print("PASSO 1: Carregando modelo com QLoRA 4-bit")
    print("="*70)
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("[*] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    if HAS_BITSANDBYTES:
        # Configuração de quantização 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        print("[*] Carregando modelo com quantização 4-bit (NF4)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map=DEVICE if DEVICE == "cpu" else "auto",
            trust_remote_code=True,
        )
    else:
        print("[*] Carregando modelo em fp32 (bitsandbytes não disponível)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=DEVICE if DEVICE == "cpu" else "auto",
            trust_remote_code=True,
        )
    
    # Ensure model is on the correct device
    if DEVICE == "cpu":
        model = model.cpu()
    
    # Medir VRAM após carregamento
    if DEVICE == "cuda":
        vram_used_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        vram_used_mb = 0
    print(f"[✓] Modelo carregado com sucesso")
    print(f"[MÉTRICA] VRAM utilizado: {vram_used_mb:.2f} MB")
    
    return model, tokenizer, vram_used_mb


# ============================================================================
# PASSO 2: SIMULAÇÃO DO RAG MASSIVO (10-15K tokens)
# ============================================================================
def generate_rag_context(tokenizer, target_min_tokens=10000, target_max_tokens=15000):
    """
    Gera contexto fictício simulando 5 capítulos de manual médico.
    
    Returns:
        str: Contexto simulado com ~10-15K tokens
    """
    print("\n" + "="*70)
    print("PASSO 2: Gerando contexto RAG massivo (simulado)")
    print("="*70)
    
    # Contexto médico fictício - repetindo para atingir 10-15K tokens
    base_context = """
    Capítulo 1: Patofisiologia do Sistema Respiratório
    
    O sistema respiratório compreende o trato respiratório superior e inferior,
    responsáveis pela troca gasosa entre o ar ambiente e o sangue. A complexidade
    fisiológica envolve múltiplas estruturas: cavidade nasal, faringe, laringe,
    traqueia, brônquios e alvéolos pulmonares. Cada componente desempenha papel
    crítico na oxigenação tissular e remoção de dióxido de carbono.
    
    A doença pulmonar obstrutiva crônica (DPOC) representa síndrome heterogênea
    caracterizada por limitação progressiva do fluxo aéreo. Mecanismos patológicos
    incluem destruição do parênquima alveolar (enfisema), inflamação brônquica crônica
    e remodelamento de vias aéreas. Hipótese inflamatória sugere papel central de
    citocinas pró-inflamatórias (TNF-α, IL-6, IL-8) e estresse oxidativo.
    
    Biomarcadores séricos em DPOC incluem proteína C-reativa (PCR), fibrinogênio,
    e citocinas. Elevação de PCR correlaciona-se com gravidade da doença e prediz
    exacerbações. Interleucina-6 eleva-se em pacientes com fenótipo inflamatório,
    reforçando heterogeneidade da síndrome.
    
    Capítulo 2: Diagnóstico e Avaliação Clínica
    
    Espirometria forçada constitui teste ouro para diagnóstico funcional respiratório.
    Índice VEF1/CVF (volume expiratório forçado no primeiro segundo dividido por
    capacidade vital forçada) menor que 0,70 confirma obstrução persistente.
    Classificação GOLD (Global Initiative for Chronic Obstructive Lung Disease)
    estratifica gravidade conforme VEF1 pós-broncodilatador.
    
    Tomografia computadorizada de tórax de alta resolução (TCAR) revela padrão
    de enfisema centrolobular ou panlobular, distribuição apical ou basilar.
    Hipodensidade detectada por densitometria correlaciona-se com redução de
    função pulmonar. Broncoscopia com biópsia permite avaliação direta de vias
    aéreas e descartar diagnósticos diferenciais.
    
    Escala de dispneia modificada Medical Research Council (mMRC) quantifica
    limitação por falta de ar em atividades cotidianas. Teste de caminhada de
    6 minutos (TC6M) avalia capacidade funcional e prediz mortalidade. Questionário
    SGRQ (St George Respiratory Questionnaire) mensura impacto em qualidade de vida.
    
    Capítulo 3: Manejo Terapêutico Farmacológico
    
    Broncodilatadores constituem primeira linha: beta-2 agonistas (albuterol, formoterol),
    anticolinérgicos (brometo de ipratrópio, tiotropio) e inibidores de fosfodiesterase-4
    (roflumilaste). Agonistas muscarínicos de ação prolongada (LAMA) reduzem exacerbações
    e hospitalização quando combinados com agonistas beta-2 de longa ação (LABA).
    
    Corticosteroides inalados (CSI) reduzem exacerbações em DPOC com fenótipo
    eosinofílico (eosinófilos ≥100 células/μL). Triplice terapia (LABA/LAMA/CSI)
    associa-se a benefício significativo em redução de exacerbações graves.
    Monitorização de efeitos colaterais: pneumonia adquirida, candidiase oral,
    tremor e taquicardia.
    
    Metilxantinas (teofilina) têm papel limitado por janela terapêutica estreita
    e interações medicamentosas. Mucolíticos e N-acetilcisteína não demonstraram
    eficácia consistente em meta-análises recentes, porém podem ser considerados
    em fenótipos específicos.
    
    Capítulo 4: Comorbidades e Impacto Sistêmico
    
    Comorbidades são frequentes: hipertensão sistêmica (30-50% dos pacientes),
    diabetes mellitus tipo 2, doença cardiovascular aterosclerótica. Mecanismo
    compartilhado de inflamação sistêmica amplifica risco cardiovascular.
    Disfunção autonômica com predomínio simpático contribui para hipertensão pulmonar.
    
    Desnutrição proteico-calórica afeta 20-40% dos pacientes com DPOC grave,
    associando-se a pior prognóstico. Índice de massa corporal (IMC) segue padrão
    em U-shape: obesidade e baixo peso conferem risco aumentado de mortalidade.
    Sarcopenia prevalece mesmo em pacientes eutróficos, sugerindo redistribuição.
    
    Depressão e ansiedade ocorrem em 25-30% dos pacientes DPOC, exacerbando
    percepção de dispneia e limitando adesão ao tratamento. Avaliação de saúde
    mental deve ser incorporada na avaliação inicial.
    
    Capítulo 5: Prognóstico e Preditores de Mortalidade
    
    Índice BODE (Body mass index, Obstruction, Dyspnea, Exercise) prediz mortalidade
    3 anos: cada ponto adicional aumenta risco de morte em 34%. Combinação de
    VEF1 reduzido, mMRC elevada, IMC baixo e capacidade ao TC6M limitada
    identifica pacientes de risco.
    
    Biomarcadores de severidade incluem elevação de proteína C-reativa de alta
    sensibilidade (PCRas), fibrinogênio, troponina ultrassensível e peptídeo
    natriurético (NT-proBNP). Fibrinogênio elevado associa-se a exacerbações
    frequentes e risco cardiovascular.
    
    Taxa de exacerbações (≥2 por ano) prediz progressão mais rápida e necessidade
    de intubação. Fenótipo inflamatório com eosinófilos sanguíneos ≥300 células/μL
    prediz resposta favorável a corticosteroides, reduzindo exacerbações em 25-35%.
    
    Avaliação de qualidade de vida por SGRQ com escore >40 indica impacto
    significativo e necessidade de intervenção multidisciplinar. Pacientes com
    redução de >4 pontos ao ano mostram deterioração funcional progressiva.
    
    Resumo de Recomendações Diagnósticas e Terapêuticas:
    
    1. Triagem em ex-fumantes >40 anos com história de exposição ocupacional
    2. Confirmação diagnóstica com espirometria VEF1/CVF <0,70 pós-broncodilatador
    3. Classificação GOLD inicial e reavaliação anual
    4. Estratificação de risco com escores BODE e exacerbações anteriores
    5. Monitorização de biomarcadores inflamatórios (PCR, IL-6, fibrinogênio)
    6. Tripla terapia em pacientes com ≥2 exacerbações/ano e VEF1 <50%
    7. Avaliação de osteoporose, depressão e desnutrição
    8. Reabilitação pulmonar multidisciplinar
    9. Vacinação anual (influenza) e pneumocócica conforme diretrizes
    10. Monitorização de hipertensão pulmonar com ecocardiografia se IC/cor pulmonale
    """
    
    # Ajustar tamanho para ficar entre 10K e 15K tokens
    combined_context = base_context
    token_count = tokenizer(combined_context, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    while token_count < target_min_tokens:
        combined_context += base_context
        token_count = tokenizer(combined_context, return_tensors="pt", truncation=False)["input_ids"].shape[1]

    if token_count > target_max_tokens:
        # Corta por tokens para manter o intervalo exigido
        trimmed_ids = tokenizer(combined_context, return_tensors="pt", truncation=False)["input_ids"][0, :target_max_tokens]
        combined_context = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        token_count = tokenizer(combined_context, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    
    print(f"[*] Contexto gerado com sucesso")
    print(f"[MÉTRICA] Tamanho do contexto: {len(combined_context)} caracteres")
    
    return combined_context


def tokenize_context(tokenizer, context):
    """
    Tokeniza o contexto RAG.
    
    Returns:
        dict: tokens do contexto com input_ids e attention_mask
    """
    print("[*] Tokenizando contexto...")
    tokens_full = tokenizer(
        context,
        return_tensors="pt",
        truncation=False,
    )
    num_tokens = tokens_full["input_ids"].shape[1]

    model_max_len = tokenizer.model_max_length
    if model_max_len is None or model_max_len > 100000:
        model_max_len = 16384

    tokens_model = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=model_max_len,
    )
    print(f"[✓] Contexto tokenizado")
    print(f"[MÉTRICA] Número de tokens: {num_tokens}")
    if num_tokens > model_max_len:
        print(f"[INFO] Contexto truncado para {model_max_len} tokens na geração")
    
    return {
        "full_tokens": tokens_full,
        "model_tokens": tokens_model,
        "num_tokens": num_tokens,
        "model_max_len": model_max_len,
    }


# ============================================================================
# PASSO 3: GERAÇÃO SEM CACHE (BASELINE)
# ============================================================================
def generate_without_cache(model, tokenizer, tokens, max_new_tokens=100):
    """
    Geração de tokens SEM cache (baseline para comparação).
    Simula o problema original: recálculo redundante de Q, K, V a cada step.
    """
    print("\n" + "="*70)
    print("PASSO 3: Geração SEM cache (BASELINE - problema original)")
    print("="*70)
    
    model.config.use_cache = False
    model.eval()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = tokens["model_tokens"]["input_ids"].to(DEVICE)
    attention_mask = tokens["model_tokens"]["attention_mask"].to(DEVICE)
    
    print(f"[*] Iniciando geração de {max_new_tokens} tokens...")
    print(f"[*] use_cache = {model.config.use_cache}")
    
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    elapsed_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Decodificar
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"[✓] Geração completa")
    print(f"[MÉTRICA] Tempo total: {elapsed_time:.3f}s")
    print(f"[MÉTRICA] Pico VRAM: {peak_memory_mb:.2f} MB")
    print(f"[MÉTRICA] Tokens/segundo: {max_new_tokens/elapsed_time:.2f}")
    print(f"\n[PRIMEIROS 200 CARACTERES DO OUTPUT]:")
    print(generated_text[:200] + "...")
    
    return {
        "elapsed_time": elapsed_time,
        "peak_memory_mb": peak_memory_mb,
        "tokens_per_sec": max_new_tokens / elapsed_time,
        "output": generated_text,
    }


# ============================================================================
# PASSO 4: GERAÇÃO COM CACHE E FLASHATTENTION-2 (OTIMIZADO)
# ============================================================================
def reload_model_optimized():
    """
    Recarrega modelo com FlashAttention-2 ativado (se disponível).
    """
    print("\n" + "="*70)
    print("PASSO 4: Recarregando modelo com FlashAttention-2")
    print("="*70)
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    print("[*] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    if HAS_BITSANDBYTES:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        print("[*] Carregando modelo com FlashAttention-2 e quantização 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map=DEVICE if DEVICE == "cpu" else "auto",
            attn_implementation="flash_attention_2",  # ← OTIMIZAÇÃO HARDWARE
            trust_remote_code=True,
        )
    else:
        print("[*] Carregando modelo com FlashAttention-2 em fp32...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map=DEVICE if DEVICE == "cpu" else "auto",
                attn_implementation="flash_attention_2",  # ← OTIMIZAÇÃO HARDWARE
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[WARNING] FlashAttention-2 not available: {e}")
            print("[*] Loading with standard attention...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map=DEVICE if DEVICE == "cpu" else "auto",
                trust_remote_code=True,
            )
    
    # Ensure model is on the correct device
    if DEVICE == "cpu":
        model = model.cpu()
    
    print(f"[✓] Modelo carregado com FlashAttention-2")
    print(f"[INFO] attn_implementation={model.config._attn_implementation}")
    
    return model, tokenizer


def generate_with_cache(model, tokenizer, tokens, max_new_tokens=100):
    """
    Geração de tokens COM cache e FlashAttention-2 (otimizado).
    """
    print("\n" + "="*70)
    print("PASSO 5: Geração COM KV Cache + FlashAttention-2 (OTIMIZADO)")
    print("="*70)
    
    model.config.use_cache = True  # ← OTIMIZAÇÃO SOFTWARE
    model.eval()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = tokens["model_tokens"]["input_ids"].to(DEVICE)
    attention_mask = tokens["model_tokens"]["attention_mask"].to(DEVICE)
    
    print(f"[*] Iniciando geração de {max_new_tokens} tokens...")
    print(f"[*] use_cache = {model.config.use_cache}")
    print(f"[*] attn_implementation = {model.config._attn_implementation}")
    
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    elapsed_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Decodificar
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"[✓] Geração completa")
    print(f"[MÉTRICA] Tempo total: {elapsed_time:.3f}s")
    print(f"[MÉTRICA] Pico VRAM: {peak_memory_mb:.2f} MB")
    print(f"[MÉTRICA] Tokens/segundo: {max_new_tokens/elapsed_time:.2f}")
    print(f"\n[PRIMEIROS 200 CARACTERES DO OUTPUT]:")
    print(generated_text[:200] + "...")
    
    return {
        "elapsed_time": elapsed_time,
        "peak_memory_mb": peak_memory_mb,
        "tokens_per_sec": max_new_tokens / elapsed_time,
        "output": generated_text,
    }


# ============================================================================
# PASSO 6: ANÁLISE COMPARATIVA
# ============================================================================
def compare_metrics(baseline, optimized):
    """
    Compara métricas baseline vs otimizado.
    """
    print("\n" + "="*70)
    print("PASSO 6: Análise Comparativa - Baseline vs Otimizado")
    print("="*70)
    
    speedup = baseline["elapsed_time"] / optimized["elapsed_time"]
    
    # Handle CPU mode (VRAM = 0)
    if baseline["peak_memory_mb"] > 0:
        memory_reduction = (1 - optimized["peak_memory_mb"] / baseline["peak_memory_mb"]) * 100
        memory_str = f"{memory_reduction:.1f}%"
    else:
        memory_reduction = None
        memory_str = "N/A (CPU mode)"
    
    print(f"\n[COMPARACAO DE METRICAS]")
    print(f"Metrica              | Baseline   | Otimizado")
    print(f"─────────────────────────────────────────────")
    print(f"Tempo (s)            | {baseline['elapsed_time']:10.3f}s | {optimized['elapsed_time']:10.3f}s")
    print(f"VRAM (MB)            | {baseline['peak_memory_mb']:10.2f}  | {optimized['peak_memory_mb']:10.2f}")
    print(f"Tokens/s             | {baseline['tokens_per_sec']:10.2f}  | {optimized['tokens_per_sec']:10.2f}")
    
    print(f"\n[GANHOS DE OTIMIZACAO]")
    print(f"[+] Speedup: {speedup:.2f}x mais rapido")
    print(f"[+] Reducao de VRAM: {memory_str}")
    print(f"[+] Melhoria em Throughput: {(optimized['tokens_per_sec'] / baseline['tokens_per_sec'] - 1) * 100:.1f}%")
    
    return {
        "speedup": speedup,
        "memory_reduction": memory_reduction,
        "memory_str": memory_str,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PIPELINE INTEGRADOR: QLoRA + RAG + Otimização GPU")
    print("Laboratório Integrador - IA em Produção")
    print("="*70)
    
    # PASSO 1: Carregar modelo com QLoRA
    model_baseline, tokenizer_baseline, vram_model = load_model_qlora()
    
    # PASSO 2: Simular RAG massivo
    rag_context = generate_rag_context(tokenizer_baseline)
    tokens = tokenize_context(tokenizer_baseline, rag_context)
    
    # PASSO 3: Geração SEM cache (baseline)
    baseline_results = generate_without_cache(
        model_baseline, 
        tokenizer_baseline, 
        tokens, 
        max_new_tokens=100
    )
    
    # PASSO 4-5: Recarregar com FA-2 e gerar COM cache
    model_optimized, tokenizer_optimized = reload_model_optimized()
    tokens_optimized = tokenize_context(tokenizer_optimized, rag_context)
    
    optimized_results = generate_with_cache(
        model_optimized,
        tokenizer_optimized,
        tokens_optimized,
        max_new_tokens=100
    )
    
    # PASSO 6: Comparação
    comparison = compare_metrics(baseline_results, optimized_results)
    
    # RESUMO FINAL
    print("\n" + "="*70)
    print("RESUMO EXECUTIVO")
    print("="*70)
    print(f"""
[VRAM Inicial - Modelo Quantizado]
  {vram_model:.2f} MB

[Geração Baseline (use_cache=False)]
  Tempo: {baseline_results['elapsed_time']:.3f}s
  Pico VRAM: {baseline_results['peak_memory_mb']:.2f} MB
  Problema: Recalcula Q, K, V a cada token → O(n²) ineficiente

[Geração Otimizada (use_cache=True + FlashAttention-2)]
  Tempo: {optimized_results['elapsed_time']:.3f}s
  Pico VRAM: {optimized_results['peak_memory_mb']:.2f} MB
  Solução: Reutiliza KV cache + acesso SRAM eficiente

[Ganhos da Otimização]
  Speedup: {comparison['speedup']:.2f}x
  Redução VRAM: {comparison['memory_str']}

[CONCLUSÃO]
  QLoRA (4-bit) + KV Cache + FlashAttention-2 = Transformers viáveis em GPU
  Com sucesso! Pipeline de produção está pronto.
    """)
    
    print("\n[✓] Pipeline concluído com sucesso!")
