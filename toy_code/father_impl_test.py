#!/usr/bin/python3

class Father:
    def __init__(self):
        self.name = 'father'

    def save(self):
        print("Father::save()")
        self.save_impl()

    def save_impl(self):
        raise NotImplementedError


class Son1(Father):
    def __init__(self):
        super().__init__()

    def save(self):
        super().save()

    def save_impl(self):
        print("Son1::save_impl()")


class Son2(Father):
    def __init__(self):
        super().__init__()

    def save_impl(self):
        print("Son2::save_impl()")


def main():
    son = Son1()
    son.save()

    son = Son2()
    son.save()


if __name__ == "__main__":
    main()
