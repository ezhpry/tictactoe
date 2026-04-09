from ai import MonteCarloModel
from cli import TerminalGame

# 兼容旧接口：外部若仍引用 Model / Game，不会立即断裂。
Model = MonteCarloModel
Game = TerminalGame


def main() -> None:
    # 入口只负责组装模块，不承载业务逻辑。
    game = TerminalGame(model=MonteCarloModel(epsilon=0.7, count=100000))
    # 若要重新训练模型，可取消下一行注释：
    # game.model.train()
    game.run()


if __name__ == "__main__":
    main()
