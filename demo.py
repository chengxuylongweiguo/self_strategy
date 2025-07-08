from WindPy import w
w.start()
a = w.wsi("000852.SH", "close", "2025-07-08 14:00:50", "2025-07-08 14:00:50", "Fill=Previous")
print(a)