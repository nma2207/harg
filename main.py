from view import View
from logic import Logic

if __name__ == "__main__":
    view = View()
    logic = Logic()
    view.set_logic(logic)
    logic.set_view(view)

    # загрузка мат.моделей
    logic.start()
    view.start()