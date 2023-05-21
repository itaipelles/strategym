import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from axis_and_allies_game.board import Board, Players

class GameRenderer:
    def __init__(self, board:Board) -> None:
        self.G = nx.Graph()
        self.G.add_nodes_from(board.territories_id)
        self.G.add_edges_from(board.adjacencies)
        self.G_node_pos = nx.spectral_layout(self.G)
        self.G_node_pos = nx.spring_layout(self.G, pos=self.G_node_pos)
        pass

    def render(self, board:Board, current_move, legality:bool):
        # the commented lines might be needed when we scale to more types of units. for now i print the infantries directly
        # nx.set_node_attributes(G, dict(zip(territories,obs['player1_infantry'])), name = 'p1_infantry')
        # nx.set_node_attributes(G, dict(zip(territories,obs['player2_infantry'])), name = 'p2_infantry')
        
        fig = plt.figure()
        plt.axis('off')
        plt.tight_layout()

        G_p1_pos = self.G_node_pos.copy()
        G_p2_pos = self.G_node_pos.copy()
        for i, key in (self.G_node_pos.items()):
            G_p1_pos[i] = self.G_node_pos[i] + [0, 0.15]
            G_p2_pos[i] = self.G_node_pos[i] - [0, 0.15]
        
        color_map = []
        for territory in board.territories:
            if territory.owner==Players.RUSSIA:
                color_map.append('red')
            elif territory.owner==Players.GERMANY:
                color_map.append('black')

        nx.draw_networkx(self.G, self.G_node_pos, with_labels = True, 
                node_color = color_map, node_size=500,
                font_color = "white", font_size = 15)
        nx.draw_networkx_labels(self.G,G_p1_pos, labels = dict(zip(board.territories_id,board.get_player_infantry(Players.RUSSIA))), font_color = "red")
        nx.draw_networkx_labels(self.G,G_p2_pos, labels = dict(zip(board.territories_id,board.get_player_infantry(Players.GERMANY))), font_color = "black")

        edge_labels = {tuple(sorted(x)) : 0 for x in board.adjacencies}
        for infantry_to_move, (from_territory, to_territory) in zip(current_move, board.adjacencies):
            if(to_territory < from_territory):
                edge_labels[(to_territory,from_territory)] -= infantry_to_move
            else:
                edge_labels[(from_territory,to_territory)] += infantry_to_move

        nx.draw_networkx_edge_labels(
            self.G, self.G_node_pos,
            edge_labels=edge_labels,
            font_color='green')

        fig.set_facecolor("white")
        if(not legality):
            fig.set_facecolor("gray")

        fig.canvas.draw()
        plt.show()
        return