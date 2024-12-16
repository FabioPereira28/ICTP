import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns

# Parâmetros principais
nepochs_otimizacao = 2  # Épocas reduzidas durante a otimização
nepochs_final = 20      # Épocas completas no treinamento final
larg, alt = 128, 128    # Resolução reduzida para acelerar
batch_size = 256        # Maior batch size para melhorar a eficiência
num_classes = 10

# Caminhos para treino e teste
treino = r"C:\Users\fabio\Desktop\ESTE_SEMESTRE\IC\TP\Dataset_dividido\train"
teste = r"C:\Users\fabio\Desktop\ESTE_SEMESTRE\IC\TP\Dataset_dividido\test"

# Data augmentation e normalização
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # Divisão para validação
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Conjuntos de treino, validação e teste
ds_treino = train_datagen.flow_from_directory(
    treino,
    target_size=(larg, alt),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)
ds_validacao = train_datagen.flow_from_directory(
    treino,
    target_size=(larg, alt),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)
ds_teste = test_datagen.flow_from_directory(
    teste,
    target_size=(larg, alt),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Função para criar o modelo
def criar_modelo(neuronios, dropout_rate, descongelar_camadas=0):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(larg, alt, 3))
    
    # Congelar todas as camadas
    for layer in base_model.layers:
        layer.trainable = False
    
    # Descongelar apenas algumas camadas no fine-tuning
    if descongelar_camadas > 0:
        for layer in base_model.layers[-descongelar_camadas:]:
            layer.trainable = True
    
    modelo = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(neuronios, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return modelo

# Função objetivo para otimização
contador = 0  # Para contar o número de combinações testadas
def funcao_objetivo(params):
    global contador
    contador += 1
    neuronios, dropout_rate = int(params[0]), params[1]
    print(f"\n[INFO] Testando configuração {contador}: Neurônios={neuronios}, Dropout={dropout_rate:.2f}")
    modelo = criar_modelo(neuronios, dropout_rate)
    modelo.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = modelo.fit(ds_treino, validation_data=ds_validacao, epochs=nepochs_otimizacao, verbose=1)
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"[INFO] Validação Final: {val_accuracy:.4f}")
    return -val_accuracy  # Maximizar a acurácia, por isso retorna negativo

# Ajustar limites para otimização
bounds = [(32, 128), (0.2, 0.5)]  # Neurônios e dropout ajustados

# Otimização com Differential Evolution
print("[INFO] Iniciando otimização...")
result = differential_evolution(funcao_objetivo, bounds, strategy='best1bin', maxiter=1, popsize=2, tol=0.01, seed=42)
optimal_params = result.x
print(f"\n[INFO] Total de combinações testadas: {contador}")
print(f"\n[INFO] Melhores parâmetros encontrados: Neurônios={int(optimal_params[0])}, Dropout={optimal_params[1]:.2f}")

# Criar e treinar o modelo final com mais épocas
modelo_final = criar_modelo(int(optimal_params[0]), optimal_params[1], descongelar_camadas=10)
modelo_final.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\n[INFO] Iniciando treinamento final...")
history = modelo_final.fit(ds_treino, validation_data=ds_validacao, epochs=nepochs_final, verbose=1)

# Avaliação final
print("\n[INFO] Avaliando no conjunto de teste...")
y_pred = modelo_final.predict(ds_teste)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test = np.concatenate([ds_teste[i][1] for i in range(len(ds_teste))])

# Métricas
conf_matrix = confusion_matrix(y_test, y_pred_classes)
classification_rep = classification_report(y_test, y_pred_classes, zero_division=1)
accuracy_final = accuracy_score(y_test, y_pred_classes)

# Função para plotar a matriz de confusão
def plotar_matriz_confusao(conf_matrix, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Verdadeira")
    plt.title("Matriz de Confusão")
    plt.show()

# Função para plotar a curva ROC
def plotar_curva_roc(y_test, y_pred, num_classes):
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Classe {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")  # Linha de referência
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC Multiclasse")
    plt.legend(loc="lower right")
    plt.show()

# Plotar gráficos
classes = [f"Classe {i}" for i in range(num_classes)]  # Nomes das classes
plotar_curva_roc(y_test, y_pred, num_classes)
plotar_matriz_confusao(conf_matrix, classes)

# Exibir resultados finais
print("\nMatriz de Confusão:")
print(conf_matrix)
print("\nRelatório de Classificação:")
print(classification_rep)
print(f"\nAcurácia Final: {accuracy_final:.4f}")

modelo_final.save('modeloFinal.h5')
print("\n[INFO] Modelo final salvo como 'modeloFinal.h5'")
