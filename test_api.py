#!/usr/bin/env python3
"""测试LongCat-AudioDiT API服务"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from api.app import create_app
from api.config import ConfigManager

def main():
    print("=== LongCat-AudioDiT API 服务测试 ===\n")
    
    # 检查目录结构
    print("1. 检查目录结构:")
    print(f"   output目录: {os.path.exists('output')}")
    print(f"   samples目录: {os.path.exists('samples')}")
    print(f"   assets目录: {os.path.exists('assets')}")
    
    # 创建测试应用
    print("\n2. 创建应用...")
    config = ConfigManager()
    app = create_app(config)
    
    # 创建测试客户端
    client = TestClient(app)
    
    print("\n3. 测试API端点:")
    
    # 测试根端点
    print("\n   a) 根端点 (/):")
    response = client.get('/')
    print(f"      状态码: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"      名称: {data.get('name')}")
        print(f"      版本: {data.get('version')}")
        print(f"      模型加载: {data.get('model_loaded')}")
    else:
        print(f"      错误: {response.text}")
    
    # 测试健康检查
    print("\n   b) 健康检查 (/health):")
    response = client.get('/health')
    print(f"      状态码: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"      状态: {data.get('status')}")
        print(f"      模型加载: {data.get('model_loaded')}")
    else:
        print(f"      错误: {response.text}")
    
    # 测试获取模型列表
    print("\n   c) 模型列表 (/get_models_list):")
    response = client.get('/get_models_list')
    print(f"      状态码: {response.status_code}")
    if response.status_code == 200:
        models = response.json()
        print(f"      找到 {len(models)} 个模型:")
        for model in models:
            print(f"      - {model.get('name')}: {model.get('description')}")
    else:
        print(f"      错误: {response.text}")
    
    # 测试获取文件夹
    print("\n   d) 文件夹信息 (/get_folders):")
    response = client.get('/get_folders')
    print(f"      状态码: {response.status_code}")
    if response.status_code == 200:
        folders = response.json()
        print(f"      输出目录: {folders.get('output_folder')}")
        print(f"      样本目录: {folders.get('samples_folder')}")
    else:
        print(f"      错误: {response.text}")
    
    # 测试获取语言列表
    print("\n   e) 语言列表 (/languages):")
    response = client.get('/languages')
    print(f"      状态码: {response.status_code}")
    if response.status_code == 200:
        languages = response.json()
        print(f"      找到 {len(languages)} 种语言")
        supported = [lang for lang in languages if lang.get('supported')]
        print(f"      其中 {len(supported)} 种受支持")
    else:
        print(f"      错误: {response.text}")
    
    # 测试获取说话人列表
    print("\n   f) 说话人列表 (/speakers):")
    response = client.get('/speakers')
    print(f"      状态码: {response.status_code}")
    if response.status_code == 200:
        speakers = response.json()
        print(f"      找到 {len(speakers)} 个说话人")
        if speakers:
            for speaker in speakers[:3]:  # 最多显示3个
                print(f"      - {speaker.get('name')}")
    else:
        print(f"      错误: {response.text}")
    
    print("\n=== 测试完成 ===")
    print("\n要启动完整的API服务器，请运行:")
    print("  python api_server.py")
    print("\n然后访问:")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  API文档: http://localhost:8000/redoc")

if __name__ == "__main__":
    main()