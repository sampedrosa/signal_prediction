import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from dtaidistance import dtw as dtw_lib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from tqdm import tqdm

def start(results, confidence=0.5):
    if results.pose_landmarks and results.face_landmarks:
        visibility = sum([landmark.visibility for landmark in results.pose_landmarks.landmark[:25]])/25
        if visibility > confidence:
            if results.right_hand_landmarks or results.left_hand_landmarks:
                rh_line = (results.pose_landmarks.landmark[18].y + results.pose_landmarks.landmark[20].y + results.pose_landmarks.landmark[22].y)/3
                lh_line = (results.pose_landmarks.landmark[17].y + results.pose_landmarks.landmark[19].y + results.pose_landmarks.landmark[21].y)/3
                hip_line = (results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y)/2
                if hip_line > rh_line or hip_line > lh_line:
                    return True
        else:
            print(f'Baixa visiblidade do pose: {visibility}')
    return False

def get_landmarks(results, fclms=[10, 67, 297, 54, 284, 162, 389, 234, 454, 132, 361, 58, 288, 136, 365, 149, 378, 148, 377]):
    landmarks = {}
    for i, landmark in enumerate(results.pose_landmarks.landmark[:25]):
        landmarks[f'PS{i}x'] = landmark.x
        landmarks[f'PS{i}y'] = landmark.y
    for i, landmark in enumerate([results.face_landmarks.landmark[fclm] for fclm in fclms]):
        landmarks[f'FC{i}x'] = landmark.x
        landmarks[f'FC{i}y'] = landmark.y
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            landmarks[f'RH{i}x'] = landmark.x
            landmarks[f'RH{i}y'] = landmark.y
    else:
        for i in range(21):
            landmarks[f'RH{i}x'] = None
            landmarks[f'RH{i}y'] = None
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            landmarks[f'LH{i}x'] = landmark.x
            landmarks[f'LH{i}y'] = landmark.y
    else:
        for i in range(21):
            landmarks[f'LH{i}x'] = None
            landmarks[f'LH{i}y'] = None
    if len(landmarks) != (134+(len(fclms)*2)):
        raise ValueError(f'Erro na extração dos landmarks: {len(landmarks)}')
    else:
        return landmarks

def centralize(landmarks:dict):
    center = {}
    center['x'] = (landmarks['PS0x'] + landmarks['PS9x'] + landmarks['PS10x'])/3 - 0.5
    center['y'] = (landmarks['PS0y'] + landmarks['PS9y'] + landmarks['PS10y'])/3 - 0.5
    for lm in landmarks:
        if landmarks[lm]:
            if 'x' in lm:
                landmarks[lm] -= center['x']
            elif 'y' in lm:
                landmarks[lm] -= center['y']
            else:
                raise ValueError(f'Nome de landmark inválido: {lm}')
    return landmarks

def cleanse(map_landmarks, threshold=0.8):
    df = pd.DataFrame(map_landmarks)
    for col in df.columns:
        if df[col].isnull().mean() >= threshold:
            df[col] = None
    df = df.astype(float).interpolate(method='linear').ffill().bfill()
    return df

def mapping(path:str, resize:int=1, confidence=0.7):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Erro ao abrir o vídeo.")
    holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2, smooth_landmarks=False, min_detection_confidence=confidence, min_tracking_confidence=confidence)
    map_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width//resize, height//resize))
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if start(results):
            landmarks = get_landmarks(results)
            map_landmarks.append(centralize(landmarks))
    holistic.close()
    cap.release()
    cv2.destroyAllWindows()
    return cleanse(map_landmarks)

def video_landmarks(signal:pd.DataFrame, path:str, fps:int=30, frame_size:tuple=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    pose, righthand, lefthand = signal.iloc[:,:-84], signal.iloc[:,-84:-42], signal.iloc[:,-42:]
    for idx in range(len(signal)):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8) # Fundo preto
        if not righthand.isnull().all().all():
            for col in righthand.columns:
                if 'x' in col:
                    x = int(righthand.iloc[idx][col] * frame_size[0])
                    y = int(righthand.iloc[idx][col.replace('x', 'y')] * frame_size[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Verde
        if not lefthand.isnull().all().all():
            for col in lefthand.columns:
                if 'x' in col:
                    x = int(lefthand.iloc[idx][col] * frame_size[0])
                    y = int(lefthand.iloc[idx][col.replace('x', 'y')] * frame_size[1])
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Vermelho
        for col in pose.columns:
            if 'x' in col:
                x = int(pose.iloc[idx][col] * frame_size[0])
                y = int(pose.iloc[idx][col.replace('x', 'y')] * frame_size[1])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Azul
        video_writer.write(frame)
    video_writer.release()

def record(signal:str, samples:int, path:str='DATA/DATASET', frame_size=(1280, 720), fps=30):
    os.makedirs(f'{path}/{signal}', exist_ok=True)
    file_size = len(os.listdir(f'{path}/{signal}'))
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False
    i = 1
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF
        ret, frame = cap.read()
        if not ret or key == ord('q') or i > samples:
            break
        if key == 32: # SPACE
            if not recording:
                video = f'{signal}{str(i+file_size)}.mp4'
                out = cv2.VideoWriter(f"{path}/{signal}/{video}", fourcc, fps, (frame_size[0], frame_size[1]))
                recording = True
                print("Gravação iniciada...")
        elif key == 13:  # ENTER
            if recording:
                recording = False
                out.release()
                i += 1
                print(f"Gravação {video} salva")
        elif key == 27:  # ESC
            if recording:
                recording = False
                out.release()
                os.remove(f"{path}/{signal}/{video}")
                print(f"Gravação {video} cancelada.")
        if recording and out is not None:
            out.write(frame)
        color = (0, 0, 255) if recording else (255, 255, 255)
        cv2.rectangle(frame, (0, 0), (frame_size[0]-1, frame_size[1]-1), color, 5)
        cv2.imshow('Webcam', frame)
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows

def get_input(*signals, path="INPUT/LANDMARKS", max=100, min=10):
    input = {}
    gloss = pd.read_csv('glossary.csv')
    signals = signals if signals else gloss['SIGNAL'].values
    for signal in signals:
        dfs = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.parquet') and signal in file:
                    df = pd.read_parquet(os.path.join(root, file)).fillna(0)
                    if len(df) > max:
                        df = pd.DataFrame({column: np.interp(np.linspace(0, 1, max), np.linspace(0, 1, len(df)), df[column].values) for column in df.columns})
                    if len(df) > min:
                        dfs.append(df)
        input[signal] = dfs
    return input

def label_encoder(labels, path='glossary.csv'):
    gloss = pd.read_csv(path)
    return [int(gloss.loc[gloss['SIGNAL'] == label, 'ID'].values[0]) for label in labels]

def padding(input):
    X, y = [], []
    length = max([max([df.shape[0] for df in dfs]) for dfs in input.values()])
    for label, dfs in input.items():
        for df in dfs:
            padded_data = np.pad(df.values, ((0, length - df.shape[0]), (0, 0)), 'constant')
            X.append(padded_data)
            y.append(label)        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(label_encoder(y)), dtype=torch.long)
    return X, y

def interpolate(input):
    X, y = [], []
    length = int(round(np.mean([len(df) for dfs in input.values() for df in dfs])))
    for label, dfs in input.items():
        for df in dfs:
            if len(df) == length:
                X.append(df.values)
            else:
                X.append(pd.DataFrame({column: np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(df)), df[column].values) for column in df.columns}).values)
            y.append(label)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(label_encoder(y)), dtype=torch.long)
    return X, y

def dtw(input):
    X, y = [], []
    length = int(round(np.mean([len(df) for dfs in input.values() for df in dfs])))
    for label, dfs in input.items():
        for df in dfs:
            X.append(np.array([[df.values[i, col] if i < len(df.values) else df.values[-1, col] for i, _ in dtw_lib.warping_path(np.linspace(0, 1, len(df.values)), np.linspace(0, 1, length))] for col in range(df.values.shape[1])]).T[:length])
            y.append(label)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(label_encoder(y)), dtype=torch.long)
    return X, y

class TrainingBar():
    def __init__(self, model, train_size, val_size, epochs):
        self.model = model
        self.train_size = train_size
        self.val_size = val_size
        self.epochs = epochs
        self.bar = tqdm(total=train_size, dynamic_ncols=True, position=0, leave=False)
    def reset(self, epoch):
        self.bar.set_description(f"Epoch {epoch + 1}/{self.epochs}")
        self.bar.reset()
    def update(self):
        metrics = {}
        metrics['ACC(train)'] = f"{self.model.train_accuracy[-1]:.4f}"
        metrics['ACC(val)'] = f"{self.model.val_accuracy[-1]:.4f}"
        metrics['LOSS(train)'] = f"{self.model.train_loss[-1]:.4f}"
        metrics['LOSS(val)'] = f"{self.model.val_loss[-1]:.4f}"
        metrics['DIFF'] = f"{(self.model.train_accuracy[-1]-self.model.train_accuracy[-2]+self.model.val_accuracy[-1]-self.model.val_accuracy[-2])/2:.4f}"
        self.bar.set_postfix(metrics)
        self.bar.update(1)
    def display(self):
        self.update()
        print(self.bar)

class SignActionDataset(Dataset):
    def __init__(self, input, method:str):
        self.X, self.y = method(input)
        self.shape = tuple(self.X.shape)
        self.labels = list(set(np.array(self.y)))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTM(nn.Module):
    def __init__(self, dataset:Dataset, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=dataset.shape[2], hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size//2)
        self.fc = nn.Linear(hidden_size//2, len(dataset.labels))
        self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy = [0, 0], [0, 0], [0, 0], [0, 0]

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        last_output = self.layer_norm(last_output)
        last_output = torch.relu(self.dense(last_output))
        last_output = self.dropout(last_output)
        return self.fc(last_output)

    def validation(self, max, threshold):
        if self.train_accuracy[-1] >= max and self.val_accuracy[-1] >= max:
            return True
        return np.var(self.train_accuracy[-5:]) <= threshold

    def graph(self, save_name=''):
        tacc, vacc = self.train_accuracy, self.train_accuracy
        tlos, vlos = [i if (i <= 1.0 and i != 0.0) else 1.0 for i in self.train_loss], [i if (i <= 1.0 and i != 0.0) else 1.0 for i in self.val_loss]
        plt.figure(figsize=(14, 7))
        plt.plot(range(1, len(tacc) + 1), tacc, label='Acurácia de Treino')
        plt.plot(range(1, len(vacc) + 1), vacc, label='Acurácia de Validação')
        plt.plot(range(1, len(tlos) + 1), tlos, label='Perda de Treino')
        plt.plot(range(1, len(vlos) + 1), vlos, label='Perda de Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Valor')
        plt.title('Métricas por Épocas')
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.7), fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True)
        if save_name:
            plt.savefig(f'OUTPUT/GRAPHS/{save_name}.png', format='png', dpi=500)
        plt.show()

    def save_model(self, name: str):
        torch.save(self, f'OUTPUT/MODELS/{name}.pth')

    def train_model(self, epochs, batch_size, learning_rate, weight_decay, max=0.99, threshold=1e-10, train_ratio=0.8):
        train_dataset, val_dataset = random_split(self.dataset, [int(train_ratio * len(self.dataset)), len(self.dataset) - int(train_ratio * len(self.dataset))])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        bar = TrainingBar(model=self, train_size=len(train_loader), val_size=len(val_loader), epochs=epochs)
        for epoch in range(epochs):
            bar.reset(epoch)
            self.train()
            sum_loss = 0
            all_train_preds, all_train_labels, all_val_preds, all_val_labels = [], [], [], []
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(batch_labels.cpu().numpy())
                sum_loss += loss.item()
                bar.update()
            sum_loss = 0
            self.eval()
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    outputs = self(batch_data)
                    loss = criterion(outputs, batch_labels)
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(batch_labels.cpu().numpy())
            self.train_loss.append(sum_loss / len(train_loader))
            self.val_loss.append(sum_loss / len(val_loader))
            self.train_accuracy.append(accuracy_score(all_train_labels, all_train_preds))
            self.val_accuracy.append(accuracy_score(all_val_labels, all_val_preds))
            if self.validation(max, threshold):
                break
        bar.display()

class SignActionTransformer(nn.Module):
    def __init__(self, dataset:Dataset, heads, layers, dim_feedforward, dropout):
        super(SignActionTransformer, self).__init__()
        self.dataset = dataset
        self.heads = heads
        self.layers = layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.embedding = nn.Linear(dataset.shape[2], dataset.shape[2])
        self.positional_embedding = nn.Parameter(torch.rand(dataset.shape[1], dataset.shape[2]))
        transformer_layer = nn.TransformerEncoderLayer(d_model=dataset.shape[2], nhead=heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=layers)
        self.feedforward = nn.Sequential(nn.Linear(dataset.shape[2], dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, dataset.shape[2]))
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dataset.shape[2], num_heads=heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(dataset.shape[2])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dataset.shape[2], len(dataset.labels))
        self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy = [0, 0], [0, 0], [0, 0], [0, 0]

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_embedding.unsqueeze(0).expand(x.size(0), -1, -1)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        att, _ = self.multihead_attention(x, x, x)
        att = self.feedforward(att)
        x = x + att
        x = x.sum(dim=1)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return self.fc(x)
    
    def validation(self, max, threshold):
        if self.train_accuracy[-1] >= max and self.val_accuracy[-1] >= max:
            return True
        if all(v < 0.9 for v in self.train_accuracy) and len(self.train_accuracy) > 20:
            return True
        return np.var(self.train_accuracy[-5:]) <= threshold
    
    def graph(self, save_name=''):
        tacc, vacc = self.train_accuracy, self.train_accuracy
        tlos, vlos = [i if (i <= 1.0 and i != 0.0) else 1.0 for i in self.train_loss], [i if (i <= 1.0 and i != 0.0) else 1.0 for i in self.val_loss]
        plt.figure(figsize=(14, 7))
        plt.plot(range(1, len(tacc) + 1), tacc, label='Acurácia de Treino')
        plt.plot(range(1, len(vacc) + 1), vacc, label='Acurácia de Validação')
        plt.plot(range(1, len(tlos) + 1), tlos, label='Perda de Treino')
        plt.plot(range(1, len(vlos) + 1), vlos, label='Perda de Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Valor')
        plt.title('Métricas por Épocas')
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.7), fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True)
        if save_name:
            plt.savefig(f'OUTPUT/GRAPHS/{save_name}.png', format='png', dpi=500)
        plt.show()
    
    def save_model(self, name:str):
        torch.save(self, f'OUTPUT/MODELS/{name}.pth')

    def train_model(self, epochs, batch_size, learning_rate, weight_decay, max=0.99, threshold=1e-6, train_ratio=0.8):
        train_dataset, val_dataset = random_split(self.dataset, [int(train_ratio * len(self.dataset)), len(self.dataset) - int(train_ratio * len(self.dataset))])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        bar = TrainingBar(model=self, train_size=len(train_loader), val_size=len(val_loader), epochs=epochs)
        for epoch in range(epochs):
            bar.reset(epoch)
            self.train()
            sum_loss = 0
            all_train_preds, all_train_labels, all_val_preds, all_val_labels = [], [], [], []
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(batch_labels.cpu().numpy())
                sum_loss += loss.item()
                bar.update()
            sum_loss = 0
            self.eval()
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    outputs = self(batch_data)
                    loss = criterion(outputs, batch_labels)
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(batch_labels.cpu().numpy())
            self.train_loss.append(sum_loss / len(train_loader))
            self.val_loss.append(sum_loss / len(val_loader))
            self.train_accuracy.append(accuracy_score(all_train_labels, all_train_preds))
            self.val_accuracy.append(accuracy_score(all_val_labels, all_val_preds))
            if self.validation(max, threshold):
                break
        bar.display()