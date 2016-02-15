from NumberGuesser import NumberGuesser

def given_no_information_when_asked_to_guess_test():
    number_guesser = NumberGuesser()
    result = number_guesser.guess()
    assert result is None, "Then it should provide no result."