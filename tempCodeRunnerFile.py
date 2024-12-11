# วาด Decision Regions และ Boundary
plt.figure(figsize=(8, 6))
plt.contourf(x1_grid, x2_grid, g_values, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha= 0.5)
plt.contour(x1_grid, x2_grid, g_values, levels=[0], colors='black', linewidths=2)
plt.scatter(x1[:, 0], x1[:, 1], color='purple', label='Class 1', edgecolor='k', alpha=0.8)
plt.scatter(x2[:, 0], x2[:, 1], color='yellow', label='Class 2', edgecolor='k', alpha=0.8)