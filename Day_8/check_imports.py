modules = ['pandas','numpy','matplotlib','sklearn','seaborn','pickle']
for m in modules:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: ERROR: {e}")
