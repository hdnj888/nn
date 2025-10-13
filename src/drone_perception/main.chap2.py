import numpy as np
import matplotlib.pyplot as plt
import os

def create_sample_data():
    """创建完整的示例数据文件"""
    print("正在创建示例数据文件...")
    
    # 创建更丰富的训练数据：y = 2x + 1 + 噪声
    np.random.seed(42)
    x_train = np.linspace(0, 25, 100)
    # 添加一些非线性成分使问题更有趣
    y_train = 2 * x_train + 1 + 0.5 * np.sin(x_train) + np.random.normal(0, 1, len(x_train))
    
    # 创建测试数据（无噪声）
    x_test = np.linspace(0, 25, 50)
    y_test = 2 * x_test + 1 + 0.5 * np.sin(x_test)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath("train.txt")) or ".", exist_ok=True)
    
    # 保存训练数据到文件
    with open("train.txt", "w") as f:
        for i in range(len(x_train)):
            f.write(f"{x_train[i]:.6f} {y_train[i]:.6f}\n")
    
    # 保存测试数据到文件
    with open("test.txt", "w") as f:
        for i in range(len(x_test)):
            f.write(f"{x_test[i]:.6f} {y_test[i]:.6f}\n")
    
    print("示例数据文件已成功创建！")
    print(f"训练数据: {len(x_train)} 个样本")
    print(f"测试数据: {len(x_test)} 个样本")
    
    # 显示前几行数据作为示例
    print("\n训练数据前5行示例:")
    with open("train.txt", "r") as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"第{i+1}行: {line.strip()}")
            else:
                break

def load_data(filename):
    """载入数据。"""
    print(f"正在加载数据文件: {filename}")
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"数据文件 {filename} 不存在")
    
    # 检查文件大小
    file_size = os.path.getsize(filename)
    if file_size == 0:
        raise ValueError(f"数据文件 {filename} 为空")
    
    xys = []
    line_count = 0
    with open(filename, "r") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释行
                try:
                    # 分割行并转换为浮点数
                    parts = line.split()
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        xys.append([x, y])
                    else:
                        print(f"警告：第{line_count}行数据不足: {line}")
                except ValueError as e:
                    print(f"警告：第{line_count}行解析错误: {line} - {e}")
    
    if not xys:
        raise ValueError(f"文件 {filename} 中没有有效数据")
    
    print(f"从 {filename} 成功加载 {len(xys)} 个数据点")
    
    # 转置数据
    xs, ys = zip(*xys)
    return np.asarray(xs), np.asarray(ys)

def identity_basis(x):
    """恒等基函数"""
    return np.expand_dims(x, axis=1)

def multinomial_basis(x, feature_num=10):
    """多项式基函数"""
    x = np.expand_dims(x, axis=1)
    # 包括从1次到feature_num次的所有多项式
    features = [x**i for i in range(1, feature_num + 1)]
    return np.concatenate(features, axis=1)

def gaussian_basis(x, feature_num=10):
    """高斯基函数"""
    centers = np.linspace(0, 25, feature_num)
    sigma = 25 / feature_num
    return np.exp(-0.5 * ((x[:, np.newaxis] - centers) / sigma) ** 2)

def gradient_descent(phi, y, lr=0.01, epochs=1000):
    """梯度下降算法"""
    w = np.zeros(phi.shape[1])
    
    print("梯度下降训练中...")
    for epoch in range(epochs):
        y_pred = phi @ w
        error = y - y_pred
        gradient = -2 * phi.T @ error / len(y)
        w -= lr * gradient
        
        # 每200次迭代打印损失
        if epoch % 200 == 0:
            loss = np.mean(error ** 2)
            print(f"迭代 {epoch:4d}, 损失: {loss:.6f}")
    
    final_loss = np.mean((y - phi @ w) ** 2)
    print(f"最终损失: {final_loss:.6f}")
    return w

def main(x_train, y_train, use_gradient_descent=False, basis_func=None, basis_name="恒等基"):
    """训练模型"""
    if basis_func is None:
        basis_func = identity_basis

    print(f"\n使用 {basis_name} 进行训练...")
    
    # 构造特征矩阵
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)  # 偏置项
    phi1 = basis_func(x_train)  # 基函数特征
    phi = np.concatenate([phi0, phi1], axis=1)
    
    print(f"特征矩阵形状: {phi.shape}")
    
    w_lsq = None
    w_gd = None
    
    # 最小二乘法
    print("使用最小二乘法求解...")
    w_lsq = np.dot(np.linalg.pinv(phi), y_train)
    print(f"最小二乘法权重: {w_lsq}")

    # 梯度下降法
    if use_gradient_descent:
        w_gd = gradient_descent(phi, y_train, lr=0.01, epochs=1000)
        print(f"梯度下降法权重: {w_gd}")

    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        if use_gradient_descent and w_gd is not None:
            return np.dot(phi, w_gd)
        else:
            return np.dot(phi, w_lsq)
    
    return f, w_lsq, w_gd

def evaluate(ys, ys_pred):
    """评估模型"""
    mse = np.mean((ys - ys_pred) ** 2)
    std = np.sqrt(mse)
    return std, mse

def plot_results(x_train, y_train, x_test, y_test, y_test_pred, title):
    """绘制结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_train, y_train, "ro", markersize=4, alpha=0.7, label="训练数据")
    plt.plot(x_test, y_test, "k-", linewidth=2, label="真实值")
    plt.plot(x_test, y_test_pred, "b--", linewidth=2, label="预测值")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    print("=== 线性回归演示程序 ===")
    
    # 创建示例数据
    create_sample_data()
    
    try:
        # 加载数据
        x_train, y_train = load_data("train.txt")
        x_test, y_test = load_data("test.txt")
        
        print(f"\n数据统计:")
        print(f"训练数据: {len(x_train)} 个样本")
        print(f"测试数据: {len(x_test)} 个样本")
        print(f"x范围: [{x_train.min():.2f}, {x_train.max():.2f}]")
        print(f"y范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        # 测试不同的基函数（只使用最小二乘法以简化演示）
        basis_functions = [
            ("恒等基函数", identity_basis),
            ("多项式基函数(3次)", lambda x: multinomial_basis(x, 3)),
            ("高斯基函数(5个)", lambda x: gaussian_basis(x, 5))
        ]
        
        for name, basis_func in basis_functions:
            print(f"\n{'='*50}")
            print(f"测试: {name}")
            print(f"{'='*50}")
            
            # 使用最小二乘法
            f, w_lsq, _ = main(x_train, y_train, 
                              use_gradient_descent=False, 
                              basis_func=basis_func,
                              basis_name=name)
            
            # 训练集评估
            y_train_pred = f(x_train)
            train_std, train_mse = evaluate(y_train, y_train_pred)
            print(f"训练集 - 标准差: {train_std:.4f}, MSE: {train_mse:.4f}")
            
            # 测试集评估
            y_test_pred = f(x_test)
            test_std, test_mse = evaluate(y_test, y_test_pred)
            print(f"测试集 - 标准差: {test_std:.4f}, MSE: {test_mse:.4f}")
            
            # 绘制结果
            plot_results(x_train, y_train, x_test, y_test, y_test_pred, 
                        f"{name} - 预测结果")
        
        # 单独演示梯度下降法
        print(f"\n{'='*50}")
        print("梯度下降法演示（使用恒等基函数）")
        print(f"{'='*50}")
        
        f_gd, _, w_gd = main(x_train, y_train, 
                            use_gradient_descent=True, 
                            basis_func=identity_basis,
                            basis_name="恒等基函数(梯度下降)")
        
        y_test_pred_gd = f_gd(x_test)
        test_std_gd, test_mse_gd = evaluate(y_test, y_test_pred_gd)
        print(f"梯度下降法测试集 - 标准差: {test_std_gd:.4f}, MSE: {test_mse_gd:.4f}")
        
        plot_results(x_train, y_train, x_test, y_test, y_test_pred_gd,
                    "梯度下降法 - 预测结果")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()