import jax

def main():
    pass

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument()

    if len(jax.devices("cuda")) > 0:
        print("CUDA-capable device available...")
        main()
    else:
        print("No CUDA-capable device available.")