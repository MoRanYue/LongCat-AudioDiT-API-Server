# This repository was deprecated, please check out [LocalSotaTalk](https://github.com/MoRanYue/LocalSotaTalk).

# LongCat-AudioDiT API 服务

基于 FastAPI 的 LongCat-AudioDiT TTS 模型 API 服务，提供完整的 RESTful 接口。

**此项目的RESTful API与[daswer123/xtts-api-server](https://github.com/daswer123/xtts-api-server)完全相同，目的便是能够于[SillyTavern](https://github.com/SillyTavern/SillyTavern)中使用LongCat-AudioDiT作为TTS后端。**

**API of this project is completely same with [daswer123/xtts-api-server](https://github.com/daswer123/xtts-api-server), which is to use LongCat-AudioDiT as its TTS backend in [SillyTavern](https://github.com/SillyTavern/SillyTavern) on purpose.**

> [!warning]
> 此项目最初使用人工智能构建，目前处于测试版。
> This project was built by AI in first, which is in testing now.

## 特性

- 完整的 RESTful API，符合 OpenAPI 规范
- 支持普通文本转语音和语音克隆
- 支持模型动态切换（1B 和 3.5B 模型）
- 可配置的输出目录和样本目录
- 提供 Web 界面 (Swagger UI) 用于 API 测试
- 支持 CORS，可跨域调用

## 安装依赖（Install Dependencies）

### 基础依赖
```bash
pip install -r requirements.txt
```

### API 额外依赖
```bash
pip install -r requirements_api.txt
```

或者一次性安装所有依赖：
```bash
pip install transformers>=5.3.0 torch>=2.0.0 torchaudio>=2.0.0 safetensors>=0.4.0 librosa>=0.10.0 soundfile>=0.12.0 numpy>=1.24.0 einops>=0.8.0 fastapi>=0.104.0 uvicorn[standard]>=0.24.0 python-multipart>=0.0.6 pydantic>=2.5.0
```

## 快速开始

### 1. 启动 API 服务器
```bash
# 使用默认配置
# Default configuration
python api_server.py

# 指定端口和主机
# Specify port and host
python api_server.py --host 127.0.0.1 --port 8080

# 使用 3.5B 模型
# Use 3.5B variant
python api_server.py --model-dir meituan-longcat/LongCat-AudioDiT-3.5B

# 指定输出目录
# Specify output directory
python api_server.py --output-dir ./my_output --samples-dir ./my_samples
```

### 2. 访问 Web 界面
启动后访问以下地址：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

### 3. 设置SillyTavern（Configure SillyTavern）
进入“扩展程序”标签页，展开“TTS”扩展设置，设置“选择文本转语音服务商”为“XTTSv2”，并指定“Provider Endpoint”为先前开启的主机与端口即可（例如`http://127.0.0.1:8080`）。

Enter "Extensions" tab, expand "TTS" settings, select "TTS Provider" to "XTTSv2", then specify "Provider Endpoint" to the host and port you opened (For example: `http://127.0.0.1:8080`).

> [!attention]
> 所有XTTS相关选项、语言与流式传输均不可用，仅可设置角色将克隆的源音频。
>
> All options belonging to XTTS, language and streaming are unavailable, but you can specify origin audio for characters.

## API 端点

### 信息获取端点
| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/speakers` | 获取可用说话人列表（样本文件夹中的音频文件） |
| GET | `/languages` | 获取支持的语言列表 |
| GET | `/get_folders` | 获取当前文件夹配置 |
| GET | `/get_models_list` | 获取可用模型列表 |
| GET | `/get_tts_settings` | 获取当前 TTS 设置 |
| GET | `/sample/{file_name}` | 获取样本音频文件 |

### 配置端点
| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/set_output` | 设置输出文件夹 |
| POST | `/set_speaker_folder` | 设置说话人文件夹 |
| POST | `/switch_model` | 切换当前模型 |
| POST | `/set_tts_settings` | 设置 TTS 参数 |

### TTS 端点
| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/tts_stream` | 流式 TTS（实际返回完整音频） |
| POST | `/tts_to_audio/` | 合成语音并返回音频数据 |
| POST | `/tts_to_file` | 合成语音并保存到文件 |

## API 使用示例

### 1. 获取系统信息
```bash
curl http://localhost:8000/
```

### 2. 获取可用模型
```bash
curl http://localhost:8000/get_models_list
```

### 3. 文本转语音（返回音频）
```bash
curl -X POST "http://localhost:8000/tts_to_audio/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天天气真好",
    "speaker_wav": "assets/prompt.wav",
    "language": "zh-CN"
  }' \
  --output output.wav
```

### 4. 文本转语音（保存到文件）
```bash
curl -X POST "http://localhost:8000/tts_to_file" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天天气真好",
    "speaker_wav": "assets/prompt.wav",
    "language": "zh-CN",
    "file_name_or_path": "my_audio.wav"
  }'
```

### 5. 切换模型
```bash
curl -X POST "http://localhost:8000/switch_model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "LongCat-AudioDiT-3.5B"}'
```

## 文件夹结构

```
LongCat-AudioDiT/
├── api/                    # API 服务代码
│   ├── __init__.py
│   ├── app.py             # FastAPI 应用
│   ├── config.py          # 配置管理
│   ├── models.py          # 模型管理
│   ├── schemas.py         # Pydantic 模型
│   └── routers/           # 路由模块
│       ├── __init__.py
│       ├── config.py      # 配置端点
│       └── tts.py         # TTS 端点
├── api_server.py          # 服务器启动脚本
├── requirements_api.txt   # API 额外依赖
├── output/                # 输出目录（默认）
├── samples/               # 样本目录（默认）
└── assets/                # 项目资源
    └── prompt.wav        # 示例提示音频
```

## 配置说明

### 命令行参数
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务器主机地址 |
| `--port` | `8000` | 服务器端口 |
| `--model-dir` | `meituan-longcat/LongCat-AudioDiT-1B` | 模型路径或 HF ID |
| `--output-dir` | `./output` | 输出目录 |
| `--samples-dir` | `./samples` | 样本目录 |
| `--log-level` | `INFO` | 日志级别 |
| `--config` | 无 | 配置文件路径（JSON） |
| `--reload` | 无 | 开发模式自动重载 |

### 配置文件
支持 JSON 配置文件，示例 `config.json`：
```json
{
  "output_dir": "./output",
  "samples_dir": "./samples",
  "model_dir": "meituan-longcat/LongCat-AudioDiT-1B",
  "host": "0.0.0.0",
  "port": 8000,
  "log_level": "INFO",
  "tts_steps": 16,
  "tts_guidance_strength": 4.0,
  "tts_guidance_method": "cfg"
}
```

## Python 客户端示例

```python
import requests
import json

# 基础配置
BASE_URL = "http://localhost:8000"

# 1. 获取系统状态
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# 2. 合成语音
tts_data = {
    "text": "今天天气真好，适合出去散步。",
    "speaker_wav": "assets/prompt.wav",  # 语音克隆用的参考音频
    "language": "zh-CN"
}

response = requests.post(
    f"{BASE_URL}/tts_to_audio/",
    json=tts_data,
    headers={"Content-Type": "application/json"}
)

# 保存音频文件
if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Audio saved to output.wav")
```

## 注意事项

1. **模型加载**：首次启动时会下载模型（如果未缓存），需要网络连接
2. **硬件要求**：需要 GPU 以获得最佳性能，CPU 也可运行但较慢
3. **内存要求**：
   - 1B 模型：约 4GB GPU 内存
   - 3.5B 模型：约 8GB GPU 内存
4. **语言支持**：模型主要针对中文优化，其他语言支持有限
5. **流式响应**：当前实现不真正支持流式，返回完整音频

## 故障排除

### 常见问题

1. **端口占用**
   ```bash
   # 指定其他端口
   python api_server.py --port 8080
   ```

2. **模型加载失败**
   - 检查网络连接
   - 检查 HuggingFace token（如果需要）
   - 检查磁盘空间

3. **音频文件不存在**
   - 确保样本文件存在于 `samples_dir` 中
   - 或提供完整的文件路径

4. **内存不足**
   - 尝试使用 1B 模型
   - 减少并发请求
   - 使用 CPU 模式

### 日志查看
启动时添加详细日志：
```bash
python api_server.py --log-level DEBUG
```

## 许可证

本 API 服务基于 LongCat-AudioDiT 项目，遵循相同的 MIT 许可证。
