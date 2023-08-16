import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.linspace(1, 70, 70)
    y = (0.9) ** (x - 9)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Teacher Forcing Ratio')
    plt.xlabel('epoch')
    plt.ylabel('TFR')
    plt.legend()
    plt.title('Teacher Forcing Ratio - Epoch Curve')
    
    fig.savefig('./tfr_curve.png')