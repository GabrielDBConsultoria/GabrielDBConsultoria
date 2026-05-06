import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
SCRIPT_PATH = INPUT_DIR / "roteiro.txt"
FINAL_OUTPUT = OUTPUT_DIR / "reuniao_final.mp3"
PAUSE_MS = 600
TTS_MODEL = "gpt-4o-mini-tts"

VOICE_MAP: Dict[str, Dict[str, str]] = {
    "Kai": {
        "voice": "alloy",
        "instructions": "Fale como facilitador de reunião de tecnologia: claro, objetivo e com ritmo natural.",
    },
    "Marco": {
        "voice": "echo",
        "instructions": "Fale como engenheiro de dados espontâneo, técnico e energético.",
    },
    "Rita": {
        "voice": "nova",
        "instructions": "Fale como tech lead analítica, calma, firme e organizada.",
    },
}

EXAMPLE_SCRIPT = """[Kai]
Fala, time. Beleza? Bom, chamei essa síncrona rápida porque a gente precisa fechar o planejamento da Sprint de Compras.

[Marco]
Cara... Kai, segura aí. Não manda essa mensagem pro Gabriel ainda não.

[Kai]
Pô, Marco, o que foi? Isso é o que tá travando a tua frente.

[Marco]
Então, bicho... Eu estava aqui escavando as conexões do Azure SQL e rodando um mapeamento local. Cara, o pipeline ERP para o Azure SQL já existe e está rodando em produção!

[Rita]
Peraí, como assim já existe? A gente não ia desenhar a arquitetura hoje?

[Marco]
Pois é! Eu achei um arquivo de status aqui no diretório do projeto. O ETL em Python foi concluído no dia 23 de abril. Ele já puxa os dados do Guardian via ODBC direto para a staging. Inclusive, processou mais de 24 mil ordens de compra. E o script do SharePoint também já está integrado via Graph API.

[Rita]
Deixa eu ver isso... Caramba, Marco, você tem razão. Olha aqui no repositório: o modelo que você criou ontem é literalmente uma duplicata. O schema real e oficial está no diretório do pipeline.

[Kai]
Espera, deixa eu entender. Então a automação diária que a Sofia estava avaliando se faria por Azure Function ou Power Automate...

[Marco]
Já tá feita, Kai! Tem um Task Scheduler configurado na VM chamado ETL BI Compras. Ele roda três vezes ao dia: às 8h, meio-dia e 5h da tarde. Tá tudo documentado nos logs da máquina.

[Rita]
Tá, então vamos recalibrar o cenário aqui. Basicamente, o que a gente achou que era o escopo da sprint já tá entregue. Criar pipeline? Check, já existe. Estruturar modelo SQL? Check, tá pronto. Carga diária? Check, rodando três vezes ao dia na VM.

[Kai]
Rapaz... Que notícia excelente. Isso muda tudo. O nosso bloqueio então não é infraestrutura de dados, é a entrega do dashboard em si.

[Rita]
Exatamente. O foco real agora é outro. A gente precisa validar se o ETL rodou hoje direitinho, conectar o Power BI Desktop ao Azure SQL e fechar o storytelling com os dados reais para publicar a Versão 2.

[Kai]
Perfeito. Já vou repriorizar a Sprint aqui no Board. Meta única: Dashboard V2 publicado no Power BI Service. Vamos direto pras tarefas.

[Kai]
Marco, você consegue matar a tarefa rápida hoje? Só entrar na VM e checar o log do dia de hoje. Se tiver falhado por algum motivo, você roda manualmente e avisa o Lucas.

[Marco]
Fechou, 15 minutinhos eu mato isso antes do Lucas começar.

[Kai]
Boa. Aí o Lucas entra hoje à noite na tarefa principal: conecta o Power BI no schema de compras, importa as medidas do arquivo DAX e monta o layout seguindo aquele blueprint de página única que já estava desenhado. A estimativa dele para isso é de 2 a 3 horas.

[Rita]
E sobre o Power Automate com o Financeiro?

[Kai]
Cara, isso aí fica como um teste pendente com eles, não vai bloquear o lançamento do dashboard. Vou deixar no radar.

[Rita]
Perfeito. Só precisamos confirmar uma coisa com o Gabriel para o Lucas conseguir rodar o processo hoje à noite.

[Kai]
É isso que eu ia falar. Vou mandar um ping para ele agora perguntando se ele já tem o arquivo do BI salvo localmente e se ele está com a senha do Azure SQL em mãos. Se ele tiver esses acessos, o Lucas já puxa ele na call hoje e mata essa conexão seguindo o checklist.

[Marco]
Fechou. Por aqui esclarecido. Vou lá olhar os logs.

[Rita]
Boa, time. Belo achado, Marco. Até mais tarde!

[Kai]
Valeu, pessoal, bom trabalho. Mandando a mensagem pro Gabriel em 3, 2, 1...
"""


def load_env() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrado. Configure o .env a partir do .env.example.")
    return api_key


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_script(path: Path) -> str:
    if not path.exists():
        path.write_text(EXAMPLE_SCRIPT, encoding="utf-8")
        print(f"[INFO] Roteiro não encontrado. Exemplo criado em: {path}")
    return path.read_text(encoding="utf-8")


def parse_dialogues(text: str) -> List[dict]:
    pattern = re.compile(r"^\[(?P<speaker>[^\]]+)\]\s*$")
    dialogues: List[dict] = []

    current_speaker = None
    buffer: List[str] = []

    def flush_buffer():
        nonlocal buffer, current_speaker
        if current_speaker and buffer:
            content = " ".join(line.strip() for line in buffer if line.strip()).strip()
            if content:
                dialogues.append(
                    {
                        "index": len(dialogues) + 1,
                        "speaker": current_speaker,
                        "text": content,
                    }
                )
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if match:
            flush_buffer()
            current_speaker = match.group("speaker").strip()
            continue

        if current_speaker and line:
            buffer.append(line)

    flush_buffer()
    return dialogues


def slugify_speaker(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return slug or "speaker"


def generate_tts_for_dialogue(client: OpenAI, dialogue: dict, output_path: Path, voice_map: Dict[str, Dict[str, str]]) -> None:
    speaker = dialogue["speaker"]
    config = voice_map.get(speaker)
    if not config:
        raise ValueError(f"Personagem '{speaker}' não está no VOICE_MAP.")

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=config["voice"],
        input=dialogue["text"],
        instructions=config["instructions"],
        response_format="mp3",
    )
    response.stream_to_file(str(output_path))


def merge_audios(audio_files: List[Path], final_output: Path, pause_ms: int) -> None:
    if not audio_files:
        raise RuntimeError("Nenhum áudio para mesclar.")

    combined = AudioSegment.silent(duration=0)
    pause = AudioSegment.silent(duration=pause_ms)

    for idx, file in enumerate(audio_files):
        segment = AudioSegment.from_file(file, format="mp3")
        combined += segment
        if idx < len(audio_files) - 1:
            combined += pause

    combined.export(final_output, format="mp3")


def main() -> None:
    print("[INFO] Iniciando geração de reunião em áudio...")
    ensure_directories()

    try:
        load_env()
    except RuntimeError as exc:
        print(f"[ERRO] {exc}")
        sys.exit(1)

    roteiro = load_script(SCRIPT_PATH)
    dialogues = parse_dialogues(roteiro)

    if not dialogues:
        print("[ERRO] Nenhum bloco de diálogo válido encontrado em input/roteiro.txt")
        sys.exit(1)

    client = OpenAI()
    generated_files: List[Path] = []

    print(f"[INFO] {len(dialogues)} falas encontradas. Gerando áudios...")
    for d in dialogues:
        filename = f"{d['index']:03d}_{slugify_speaker(d['speaker'])}.mp3"
        out_file = OUTPUT_DIR / filename
        try:
            generate_tts_for_dialogue(client, d, out_file, VOICE_MAP)
            generated_files.append(out_file)
            print(f"[OK] {filename} ({d['speaker']})")
        except Exception as exc:
            print(f"[ERRO] Falha ao gerar fala {d['index']} ({d['speaker']}): {exc}")
            sys.exit(1)

    try:
        merge_audios(generated_files, FINAL_OUTPUT, PAUSE_MS)
    except Exception as exc:
        print(f"[ERRO] Falha ao concatenar áudios: {exc}")
        sys.exit(1)

    print("\n[RESUMO]")
    print(f"- Falas processadas: {len(dialogues)}")
    print(f"- Arquivos individuais: {len(generated_files)} em {OUTPUT_DIR}")
    print(f"- Arquivo final: {FINAL_OUTPUT}")
    print(f"- Pausa entre falas: {PAUSE_MS} ms")


if __name__ == "__main__":
    main()
