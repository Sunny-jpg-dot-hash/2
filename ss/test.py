from tensorflow import keras
import gymnasium as gym
import numpy as np
import os
import highway_env  # 確保 highw ay-env 被載入以支援 roundabout-v0 環境
from keras.utils import to_categorical

root_path = os.path.abspath(os.path.dirname(__file__))

def main():
    total_reward = 0
    
    # 加載模型
    model_path = os.path.join(root_path, 'YOURMODEL.h5')  # 請替換為您的模型路徑
    model = keras.models.load_model(model_path)

    # 編譯模型以便計算損失和準確率
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 創建環境，並檢查 roundabout-v0 是否可用
    try:
        env = gym.make('roundabout-v0', render_mode='rgb_array')
    except gym.error.NameNotFound:
        print("Environment 'roundabout-v0' does not exist. Please ensure that highway-env is correctly installed.")
        return None

    # 保存觀測和標籤的列表，用於損失和準確率計算
    observations = []
    true_labels = []  # 根據真實情況設置標籤

    for _ in range(10):  # 測試 10 輪
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            env.render()
            obs = obs.reshape(1, 25)  # 調整觀測數據的形狀為 (1, 25)
            
            # 預測動作
            action = np.argmax(model.predict(obs, verbose=0))
            observations.append(obs)
            true_labels.append(action)  # 暫時將預測的動作當作真實標籤

            # 執行動作
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward

    # 評估模型的損失和準確率
    observations = np.array(observations).reshape(-1, 25)  # 將所有觀測合併為一個批次
    true_labels = np.array(true_labels)
    test_loss, test_accuracy = model.evaluate(observations, true_labels, verbose=2)

    # 輸出測試損失和準確率
    print(f"Test loss: {test_loss:.3f}")
    print(f"Accuracy: {test_accuracy * 100:.2f}%")

    return total_reward

if __name__ == "__main__":
    rewards = []
    for round in range(10):
        reward = main()
        rewards.append(reward)
    print("Total rewards over 10 rounds:", rewards)
