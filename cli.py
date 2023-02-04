def question_change() -> bool:
    """determine if user wants to change a frame"""
    while True:
        question_change = input("Change any frame? (y/n)")
        if question_change == "y":
            return True
        elif question_change == "n":
            return False


def question_frame(frame_positions_list: list[int]) -> list[int]:
    """determine which frames a user wants to change"""
    while True:
        try:
            question = input(
                f"Which frame numbers? (list position) {frame_positions_list}"
            )
            position = list(map(int, question.split()))
            return position
        except ValueError:
            print("the positions should be integers separated by spaces")
