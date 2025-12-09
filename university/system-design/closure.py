
def cum_sum():
    b = []
    def adder(a: int):
        b.append(a)
        return sum(b)
    return adder


adder = cum_sum()
print(adder(1))
print(adder(3))
