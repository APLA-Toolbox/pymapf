from pymapf.decentralized.position import Position

def test_position():
    pos = Position(2, 4)
    assert pos.x == 2 and pos.y == 4

def test_position_str():
    pos = Position(4, 5)
    assert str(pos) == "[4, 5]"
