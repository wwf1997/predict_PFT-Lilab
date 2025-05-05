# ------------------------------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

project_path = 'E:/'
df_1 = pd.read_excel(f'{project_path}/I3D-features.xlsx')
df_2 = pd.read_excel(f'{project_path}/lasso-Aview-features.xlsx')
trial_name = "I3D-Aview-CNN-model"
feature_num_1 = 8
feature_num_2 = 4

train_data_1 = df_1[df_1['set_split'] == 'train']
valid_data_1 = df_1[df_1['set_split'] == 'valid']
test_data_1 = df_1[df_1['set_split'] == 'test']
train_data_2 = df_2[df_2['set_split'] == 'train']
valid_data_2 = df_2[df_2['set_split'] == 'valid']
test_data_2 = df_2[df_2['set_split'] == 'test']

X_train_1 = train_data_1.iloc[:, feature_num_1:].values
X_train_2 = train_data_2.iloc[:, feature_num_2:].values
y_train = train_data_1.iloc[:, 3].values
X_valid_1 = valid_data_1.iloc[:, feature_num_1:].values
X_valid_2 = valid_data_2.iloc[:, feature_num_2:].values
y_valid = valid_data_1.iloc[:, 3].values
X_test_1 = test_data_1.iloc[:, feature_num_1:].values
X_test_2 = test_data_2.iloc[:, feature_num_2:].values
y_test = test_data_1.iloc[:, 3].values

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def calculate_ccc(y_true, y_pred):
    y_true = np.ravel(y_true)  # 展平为一维数组
    y_pred = np.ravel(y_pred)  # 展平为一维数组

    # 计算均值和标准差
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)

    # 计算协方差
    covariance = np.cov(y_true, y_pred)[0, 1]

    # 计算CCC
    ccc = (2 * covariance) / (std_true ** 2 + std_pred ** 2 + (mean_true - mean_pred) ** 2)

    return ccc

# 定义模型的创建函数
def create_model(weight_1=0.05):
    input1 = Input(shape=(X_train_1.shape[1],))
    x1 = Dense(1, activation='linear')(input1)

    input2 = Input(shape=(X_train_2.shape[1],))
    x2 = Dense(512, activation='relu')(input2)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dense(1, activation='linear')(x2)

    weight_2 = 1-weight_1

    output = weight_1 * x1 + weight_2 * x2

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# 使用K-Fold对训练集进行交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 存储每一折的结果
cv_results = {
    'CCC': [],
    'MAE': [],
    'MSE': [],
    'RMSE': [],
    'R2': [],
    # 'Pearson_r': []
}

# 网格搜索的范围：weight_1从0到1，步长为0.05
weight_1_values = np.arange(0, 1.01, 0.1)
best_weight_1 = None
# best_avg_mse = float('inf')
best_min_mse = float('inf')

# K-Fold交叉验证过程
for weight_1 in weight_1_values:
    print(f'\nTesting weight_1 = {weight_1}')

    fold_mse = []  # 用于存储每一折的 MSE

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_1)):
        # print(f'Fold {fold + 1}/{kf.get_n_splits()}')

        # 获取训练集和验证集
        X_train_fold_1, X_val_fold_1 = X_train_1[train_index], X_train_1[val_index]
        X_train_fold_2, X_val_fold_2 = X_train_2[train_index], X_train_2[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # 创建模型并训练
        model = create_model(weight_1=weight_1)

        # 设置早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
        )

        # 设置模型检查点回调
        checkpoint = ModelCheckpoint(
            f'{project_path}/{trial_name}_{fold + 1}.h5',  # 每折保存一个最佳模型
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )

        # 训练模型
        model.fit(
            [X_train_fold_1, X_train_fold_2], y_train_fold,
            epochs=1000,
            batch_size=32,
            validation_data=([X_valid_1, X_valid_2], y_valid),  # 使用固定的验证集
            callbacks=[checkpoint, early_stopping],
            shuffle=False,  # 保持数据顺序
            verbose=0
        )

        # 加载每折的最佳模型
        best_model = load_model(f'{project_path}/{trial_name}_{fold + 1}.h5')

        # 在验证集上评估模型
        val_pred = best_model.predict([X_valid_1, X_valid_2])

        # 计算每一折的性能指标
        val_mse = mean_squared_error(y_valid, val_pred)
        fold_mse.append(val_mse)

    # 获取最小的 MSE
    min_mse = np.min(fold_mse)
    print(f'Minimum MSE for weight_1 = {weight_1}: {min_mse:.4f}')

    if min_mse < best_min_mse:
        best_min_mse = min_mse
        best_weight_1 = weight_1

print(f'\nBest weight_1: {best_weight_1} with Minimum MSE: {best_min_mse:.4f}')

# --------------------------------------------------------------------------------------------------------
weight_1 = 0.2 #TODO: check

# 使用K-Fold对训练集进行交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 存储每一折的结果
cv_results = {
    'CCC': [],
    'MAE': [],
    'MSE': [],
    'RMSE': [],
    'R2': [],
    # 'Pearson_r': []
}

def create_model():
    input1 = Input(shape=(X_train_1.shape[1],))
    x1 = Dense(1, activation='linear')(input1)

    input2 = Input(shape=(X_train_2.shape[1],))
    x2 = Dense(512, activation='relu')(input2)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dense(1, activation='linear')(x2)

    weight_2 = 1-weight_1

    output = weight_1 * x1 + weight_2 * x2 # 0.86

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# K-Fold交叉验证过程
for fold, (train_index, val_index) in enumerate(kf.split(X_train_1)):
    print(f'Fold {fold + 1}/{kf.get_n_splits()}')

    # 获取训练集和验证集
    X_train_fold_1, X_val_fold_1 = X_train_1[train_index], X_train_1[val_index]
    X_train_fold_2, X_val_fold_2 = X_train_2[train_index], X_train_2[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # 创建模型
    model = create_model()

    # 设置早停回调
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # 设置模型检查点回调
    checkpoint = ModelCheckpoint(
        f'{project_path}/{trial_name}_{fold + 1}.h5',  # 每折保存一个最佳模型
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # 训练模型
    model.fit(
        [X_train_fold_1, X_train_fold_2], y_train_fold,
        epochs=1000,
        batch_size=32,
        validation_data=([X_valid_1, X_valid_2], y_valid),  # 使用固定的验证集
        callbacks=[checkpoint, early_stopping],
        shuffle=False  # 保持数据顺序
    )

    # 加载每折的最佳模型
    best_model = load_model(f'{project_path}/{trial_name}_{fold + 1}.h5')

    # 在验证集上评估模型
    val_pred = best_model.predict([X_valid_1, X_valid_2])

    # 计算每一折的性能指标
    val_ccc = calculate_ccc(y_valid, val_pred)  # 计算每折的CCC
    val_mae = mean_absolute_error(y_valid, val_pred)
    val_mse = mean_squared_error(y_valid, val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_valid, val_pred)
    # val_pearson_r, _ = pearsonr(y_valid, val_pred)

    # 输出每一折的结果
    print(f'Fold {fold + 1} - CCC: {val_ccc:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}')
    # 保存每一折的结果
    cv_results['CCC'].append(val_ccc)
    cv_results['MAE'].append(val_mae)
    cv_results['MSE'].append(val_mse)
    cv_results['RMSE'].append(val_rmse)
    cv_results['R2'].append(val_r2)
    # cv_results['Pearson_r'].append(val_pearson_r)

# 输出交叉验证的平均值和标准差
print("\nCross-Validation Results:")
for metric in cv_results:
    mean_val = np.mean(cv_results[metric])
    std_val = np.std(cv_results[metric])
    print(f'{metric} - Mean: {mean_val:.4f}, Std Dev: {std_val:.4f}')

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_excel(f'{project_path}/{trial_name}_cv_results.xlsx', index=False)

# --------------------------------------------------------------------------------------------------------
# 加载保存的最佳模型
best_fold = 1 # TODO: check
best_model = load_model(f'{project_path}/{trial_name}_{best_fold}.h5')

loss, mae = best_model.evaluate([X_valid_1, X_valid_2], y_valid)
print(f'Valid Mean Absolute Error: {mae}; loss: {loss}')
loss, mae = best_model.evaluate([X_test_1, X_test_2], y_test)
print(f'Test Mean Absolute Error: {mae}; loss: {loss}')

# 计算CCC
# 计算CCC的函数
def concordance_correlation_coefficient(y_true, y_pred):
    # 计算均值
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # 计算方差
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # 计算协方差
    covariance = np.cov(y_true, y_pred)[0][1]

    # 计算CCC
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

# 计算Pearson相关系数
def calculate_correlation(y_true, y_pred):
    correlation, _ = pearsonr(y_true, y_pred)
    return correlation

def calculate_sst(y_true):
    y_mean = np.mean(y_true)
    return np.sum((y_true - y_mean) ** 2)

# 计算残差平方和 SSR
def calculate_ssr(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# 计算 R²
def calculate_r2(sst, ssr):
    return 1 - (ssr / sst)
