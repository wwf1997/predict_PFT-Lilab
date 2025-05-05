# ----------------------------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
np.random.seed(42)
tf.random.set_seed(42)

project_path = 'E:/'
df = pd.read_excel(f'{project_path}/lasso-Aview-features.xlsx')
trial_name = "SAHW-lasso-Aview-CNN-model"

train_data = df[df['set_split'] == 'train']
valid_data = df[df['set_split'] == 'valid']
test_data = df[df['set_split'] == 'test']

# 分离特征和标签
X_train = train_data.iloc[:, 4:].values # 8; 4
y_train = train_data.iloc[:, 3].values
X_valid = valid_data.iloc[:, 4:].values # 8; 4
y_valid = valid_data.iloc[:, 3].values
X_test = test_data.iloc[:, 4:].values # 8; 4
y_test = test_data.iloc[:, 3].values


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
def create_model():
    input2 = Input(shape=(X_train.shape[1],))
    x2 = Dense(512, activation='relu')(input2)
    x2 = Dense(128, activation='relu')(x2)
    output = Dense(1, activation='linear')(x2)

    model = Model(inputs=input2, outputs=output)
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

# K-Fold交叉验证过程
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f'Fold {fold + 1}/{kf.get_n_splits()}')

    # 获取训练集和验证集
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
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
        X_train_fold, y_train_fold,
        epochs=1000,
        batch_size=32,
        validation_data=(X_valid, y_valid),  # 使用固定的验证集
        callbacks=[checkpoint, early_stopping],
        shuffle=False  # 保持数据顺序
    )

    # 加载每折的最佳模型
    best_model = load_model(f'{project_path}/{trial_name}_{fold + 1}.h5')

    # 在验证集上评估模型
    val_pred = best_model.predict(X_valid)

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
best_fold = 5 # TODO: check
best_model = load_model(f'{project_path}/{trial_name}_{best_fold}.h5')

# 评估最佳模型
loss, mae = best_model.evaluate(X_valid, y_valid)
print(f'Valid Mean Absolute Error: {mae}; loss: {loss}')
loss, mae = best_model.evaluate(X_test, y_test)
print(f'Test Mean Absolute Error: {mae}; loss: {loss}')


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
