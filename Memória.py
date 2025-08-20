import shutil

total, usado, livre = shutil.disk_usage("C:/")
print(f"Total: {total / (1024**3):.2f} GB")
print(f"Usado: {usado / (1024**3):.2f} GB")
print(f"Livre: {livre / (1024**3):.2f} GB")
