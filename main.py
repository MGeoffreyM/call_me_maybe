from src.parser import Parser


def main():
    file = Parser()
    print(file.read_file(
        '/home/gematura/cursus_42/tronc_commum/call_me_maybe'
        '/data/input/functions_definition.json'))


if __name__ == "__main__":
    main()
