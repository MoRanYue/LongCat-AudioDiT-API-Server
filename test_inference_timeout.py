#!/usr/bin/env python3
"""测试推理超时问题"""

import sys
import os
import time
import threading
import signal

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.config import ConfigManager
from api.models import ModelManager, InferenceService

def inference_with_timeout(timeout_seconds=30):
    """带超时的推理测试"""
    
    def run_inference():
        """实际运行推理的函数"""
        try:
            print("初始化模型管理器...")
            model_manager = ModelManager()
            
            print(f"加载模型...")
            model_manager.load_model("meituan-longcat/LongCat-AudioDiT-1B")
            
            print(f"创建推理服务...")
            inference = InferenceService(model_manager)
            
            # 准备测试音频路径
            speaker_wav = "assets/prompt.wav"
            if not os.path.exists(speaker_wav):
                # 尝试使用samples目录中的第一个文件
                samples_dir = "./samples"
                if os.path.exists(samples_dir):
                    wav_files = [f for f in os.listdir(samples_dir) if f.lower().endswith('.wav')]
                    if wav_files:
                        speaker_wav = os.path.join(samples_dir, wav_files[0])
                    else:
                        speaker_wav = None
            
            print(f"运行推理...")
            print(f"文本: '测试音频生成'")
            print(f"说话人音频: {speaker_wav}")
            
            start_time = time.time()
            
            audio, sr = inference.synthesize(
                text="测试音频生成",
                speaker_wav=speaker_wav,
                prompt_text=None,
                steps=4,  # 使用较少的步骤加速测试
                guidance_strength=4.0,
                guidance_method="cfg",
                seed=1024
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"推理成功！")
            print(f"音频形状: {audio.shape}")
            print(f"采样率: {sr}")
            print(f"推理时间: {duration:.2f}秒")
            
            return True, f"推理成功，耗时{duration:.2f}秒"
            
        except Exception as e:
            import traceback
            print(f"推理失败: {e}")
            traceback.print_exc()
            return False, str(e)
    
    # 在单独的线程中运行推理
    result = {"success": False, "message": "超时", "thread_completed": False}
    
    def inference_thread():
        try:
            success, message = run_inference()
            result["success"] = success
            result["message"] = message
            result["thread_completed"] = True
        except Exception as e:
            result["message"] = f"线程异常: {e}"
            result["thread_completed"] = True
    
    thread = threading.Thread(target=inference_thread)
    thread.daemon = True
    thread.start()
    
    # 等待线程完成或超时
    thread.join(timeout_seconds)
    
    if not result["thread_completed"]:
        print(f"推理超时（{timeout_seconds}秒）")
        return False, f"推理超时（{timeout_seconds}秒）"
    
    return result["success"], result["message"]

if __name__ == "__main__":
    print("=== 推理超时测试 ===\n")
    
    # 先测试快速加载
    print("1. 快速测试（4步，30秒超时）...")
    success, message = inference_with_timeout(timeout_seconds=30)
    
    if success:
        print(f"✓ 测试通过: {message}")
    else:
        print(f"✗ 测试失败: {message}")
        
        # 如果超时，尝试更短的步骤
        print("\n2. 尝试更短的测试（2步，60秒超时）...")
        
        # 修改模型管理器和推理服务以使用更少的步骤
        print("初始化最小配置...")
        try:
            model_manager = ModelManager()
            model_manager.load_model("meituan-longcat/LongCat-AudioDiT-1B")
            inference = InferenceService(model_manager)
            
            # 设置默认步数为2
            inference.default_steps = 2
            
            print("运行超短推理...")
            start_time = time.time()
            
            # 尝试不使用语音提示
            audio, sr = inference.synthesize(
                text="测试",
                speaker_wav=None,
                prompt_text=None,
                steps=2,
                guidance_strength=4.0,
                guidance_method="cfg",
                seed=1024
            )
            
            end_time = time.time()
            print(f"✓ 最小测试成功！耗时: {end_time - start_time:.2f}秒")
            
        except Exception as e:
            print(f"✗ 最小测试也失败: {e}")
            import traceback
            traceback.print_exc()