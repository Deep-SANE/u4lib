import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

tasks = ['Task 1\n(Python API)', 'Task 2\n(C++ Batch API)', 'Task 3\n(C++ OpenMP)']
x = np.arange(len(tasks))
width = 0.35

c_tok_m = [650.386, 693.475, 807.737]
c_tok_s = [304.782, 230.556, 231.065]
t_tok_m = [318.884, 311.188, 291.843]
t_tok_s = [109.257, 75.098, 69.332]

c_time_m = [97.35, 117.69, 157.59]
c_time_s = [42.34, 24.53, 69.85]
t_time_m = [62.04, 77.24, 70.20]
t_time_s = [42.67, 25.88, 12.99]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)

c_color = '#5782bb'
t_color = '#db443c'
edge_color = '#222222'
err_kw = {'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5, 'ecolor': 'black'}

# Panel (a): Tokens
rects1_a = ax1.bar(x - width/2, c_tok_m, width, yerr=c_tok_s, label='Control (Zero Skills)',
                   color=c_color, edgecolor=edge_color, error_kw=err_kw)
rects2_a = ax1.bar(x + width/2, t_tok_m, width, yerr=t_tok_s, label='Treatment (Domain Skills)',
                   color=t_color, edgecolor=edge_color, error_kw=err_kw)

ax1.set_ylabel('Total Tokens (Thousands)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Token Consumption by Task (N = 10)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, fontsize=11)
ax1.set_ylim(0, 1350)
ax1.set_yticks(np.arange(0, 1301, 200))
ax1.yaxis.grid(True, linestyle='--', alpha=0.7, color='#dbdbdb')
ax1.set_axisbelow(True)
ax1.legend(loc='upper left', frameon=True, fontsize=10, edgecolor='#cccccc')

callouts_a = ['−51.0%', '−55.1%', '−63.9%']
y_pos_a = [1030, 1000, 1120]
for i, txt in enumerate(callouts_a):
    ax1.text(x[i], y_pos_a[i], txt, ha='center', va='center', fontsize=11, fontweight='bold', color='#a50f15',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#fee5d9', edgecolor='#a50f15', linewidth=1.2))

# Panel (b): Execution Time
rects1_b = ax2.bar(x - width/2, c_time_m, width, yerr=c_time_s, label='Control (Zero Skills)',
                   color=c_color, edgecolor=edge_color, error_kw=err_kw)
rects2_b = ax2.bar(x + width/2, t_time_m, width, yerr=t_time_s, label='Treatment (Domain Skills)',
                   color=t_color, edgecolor=edge_color, error_kw=err_kw)

ax2.set_ylabel('Wall-Clock Duration (Seconds)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Execution Duration by Task (N = 10)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(tasks, fontsize=11)
ax2.set_ylim(0, 280)
ax2.set_yticks(np.arange(0, 251, 50))
ax2.yaxis.grid(True, linestyle='--', alpha=0.7, color='#dbdbdb')
ax2.set_axisbelow(True)
ax2.legend(loc='upper left', frameon=True, fontsize=10, edgecolor='#cccccc')

callouts_b = ['−36.3%', '−34.4%', '−55.4%']
y_pos_b = [156, 158, 246]
for i, txt in enumerate(callouts_b):
    ax2.text(x[i], y_pos_b[i], txt, ha='center', va='center', fontsize=11, fontweight='bold', color='#a50f15',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#fee5d9', edgecolor='#a50f15', linewidth=1.2))

plt.tight_layout()
plt.savefig('u4lib/doc/fig_benchmark_summary.pdf', format='pdf', bbox_inches='tight')
plt.savefig('u4lib/doc/fig_benchmark_summary.png', format='png', dpi=300, bbox_inches='tight')
print('Successfully generated fig_benchmark_summary.pdf and fig_benchmark_summary.png')
