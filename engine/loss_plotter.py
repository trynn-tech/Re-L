import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

DB_PATH = pathlib.Path("configs/ltm.sqlite")

def plot_learning_curve():
    if not DB_PATH.exists():
        print("❌ No LTM database found. Run the agent first.")
        return

    # Connect and pull data
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT confidence, valence, updated FROM memo ORDER BY updated ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("⚠️ Database is empty. No data to plot.")
        return

    # Convert timestamp to readable format if needed
    df['updated'] = pd.to_datetime(df['updated'], unit='s')
    
    # Calculate Moving Average to see the 'Trend'
    df['trend'] = df['confidence'].rolling(window=5, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    
    # Plot Confidence (The Loss Metric)
    plt.plot(df['updated'], df['confidence'], 'o', alpha=0.3, label='Raw Confidence (Critic Score)', color='gray')
    plt.plot(df['updated'], df['trend'], '-', label='Learning Trend (Moving Avg)', color='blue', linewidth=2)
    
    # Plot Valence (User Satisfaction)
    plt.plot(df['updated'], df['valence'], '--', label='User Valence (Mood)', color='green', alpha=0.5)

    plt.title("Hegelian Engine: Stochastic Learning Curve", fontsize=14)
    plt.xlabel("Timeline of Synthesis", fontsize=12)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.ylim(-1.1, 1.1) # Valence can be negative
    plt.axhline(y=0.85, color='r', linestyle=':', label='Expert Threshold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("configs/learning_curve.png")
    print("✅ Learning curve saved to configs/learning_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()
