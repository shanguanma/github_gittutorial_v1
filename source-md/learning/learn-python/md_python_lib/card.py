
# 2020 TL@NTU Ma Duo
# refrence:https://codingpy.com/books/thinkpython2/18-inheritance.html
class Card:
    # Both suit_name and rank_name are class attribution,
    # self.suit and self.rank are instance attribution.
    suit_name = ['Clubs','Diamonds', 'Hearts', 'Spades']
    rank_name = [None, 'Ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'Jack', 'Queen', 'King']

    def __init__(self, suit=0, rank=2):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return "%s of %s" % (Card.suit_name[self.suit], Card.rank_name[self.rank])

    def __lt__(self, other):
        t1 = self.suit, self.rank
        t2 = other.suit, other.rank
        return t1 < t2

    

