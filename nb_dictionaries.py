import pandas as pd

hotel = pd.read_csv("INNHotelsGroup.csv")
print(hotel)
data = hotel.copy()

bvar = 2


def headings(data):
    headings_dictionary = {}
    for count, column in enumerate(data):
        # key='{} : {}'.format(count, column)
        key = "{}".format(count)
        value = column
        if count < 10:
            key = "0" + key
        column = {key: column}
        headings_dictionary.update(column)
    return headings_dictionary


def headings_call(data):
    return headings(data)


headings_call(data)
