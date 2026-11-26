# Runtime images

Each directory holds the Dockerfile of one Real2SimLab environment. Build them with
`r2s env build <name>` (it assembles the build context, e.g. adds the twin SDK) and list
them with `r2s env list`. They are also shipped inside the wheel, so `r2s env build` works
from an installed package. The redistributable ones are published under the same names on the
project's GitHub Container Registry: `r2s env pull <name>`, or any command that needs a
missing image, fetches them.

| Environment | Image | Content |
|---|---|---|
| `r2s-python` | `ghcr.io/real2simhub/r2s-python:1.0` | Python 3.11, NumPy, SciPy, ffmpeg, `r2s_twin` (CPU) |
| `r2s-agent-codex` | `ghcr.io/real2simhub/r2s-agent-codex:1.0` | `r2s-python` + Node 22 + Codex CLI |
| `r2s-agent-claude-code` | `ghcr.io/real2simhub/r2s-agent-claude-code:1.0` | `r2s-python` + Node 22 + Claude Code |
| `r2s-agent-gemini-cli` | `ghcr.io/real2simhub/r2s-agent-gemini-cli:1.0` | `r2s-python` + Node 22 + Gemini CLI |
| `r2s-agent-opencode` | `ghcr.io/real2simhub/r2s-agent-opencode:1.0` | `r2s-python` + Node 22 + OpenCode |
| `r2s-verifier` | `ghcr.io/real2simhub/r2s-verifier:1.0` | `r2s-python` + pydantic and tomli-w (verifiers of Harbor exports) |
| `r2s-blender-4.0` | `ghcr.io/real2simhub/r2s-blender-4.0:1.0` | `r2s-python` + Blender 4.0.2 |
| `r2s-agent-codex-blender` | `ghcr.io/real2simhub/r2s-agent-codex-blender:1.0` | `r2s-blender-4.0` + Node 22 + Codex CLI (the `r2s-agent-codex` Dockerfile on the Blender base; built locally) |
| `r2s-video2world` | `ghcr.io/real2simhub/r2s-video2world:1.0` | `r2s-python` + ManiSkill 3.0.1, SAPIEN 3.0.3, torch (CPU) and ManiSkill assets, and a MuJoCo 3.11 interpreter (`/opt/v2w-twin`): Video2World's ManiSkill and MuJoCo evaluators (needs the GPU's Vulkan; built locally) |
| `r2s-video2world-isaac` | `ghcr.io/real2simhub/r2s-video2world-isaac:1.0` | CUDA 12.8 + Python 3.11 + PyTorch 2.7 + Isaac Sim 5.1 + RoboDojo's IsaacLab and cuRobo: Video2World's RoboDojo and in-house evaluators (NVIDIA's license: built only with `R2S_ACCEPT_NVIDIA_EULA=1`, never published; `all` skips it) |
| `r2s-agent-codex-video2world` | `ghcr.io/real2simhub/r2s-agent-codex-video2world:1.0` | `r2s-video2world` + Node 22 + Codex CLI (built locally) |
| `r2s-physimcode` | `ghcr.io/real2simhub/r2s-physimcode:1.0` | `r2s-python` + Matplotlib 3.11.2: PhySimCode's 2D programs |
| `r2s-agent-codex-physimcode` | `ghcr.io/real2simhub/r2s-agent-codex-physimcode:1.0` | `r2s-physimcode` + Node 22 + Codex CLI (the `r2s-agent-codex` Dockerfile on the PhySimCode runtime; built locally) |

Behind a slow connection to PyPI or npm, pass mirrors: `r2s env build r2s-python
--pip-index-url https://... --npm-registry https://...`.
