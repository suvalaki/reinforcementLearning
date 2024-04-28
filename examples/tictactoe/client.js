
class GameState {
    constructor(websocket) {
        this.socket = websocket;
        this.socket.onmessage = this.recieveState;
        this.board = [
            [null, null, null],
            [null, null, null],
            [null, null, null],
        ];
        this.currentPlayer = 'X';
        this.winner = null;
        this.playable = true;
        this.actionPromise = null
    }

    recieveState = (event) => {
        console.log("Received data: " + event.data);
        let data = JSON.parse(event.data);
        this.board = data.board;
        this.currentPlayer = data.currentPlayer;
        this.winner = data.winner;
        this.playable = data.playable;
        if (this.actionPromise != null){
            this.actionPromise(); // resolve the promise
            this.actionPromise = null;
        }
        console.log("Game State: " + JSON.stringify(this.board) + " Current Player: " + this.currentPlayer);
    };

    sendReset = () => {
        this.socket.send(JSON.stringify({"action": "reset"}));
    }

    sendMove = (row, col) => {
        if (this.actionPromise === null) {
            return new Promise((resolve, reject) => {
                this.socket.send(JSON.stringify(
                    {
                        "action": "move", 
                        "data": {
                            "player": gameState.currentPlayer, 
                            "row": row, 
                            "col": col
                        }
                    }));
                this.actionPromise = resolve;
            })
        }
        let emptyPromise = Promise.resolve();
        return emptyPromise;
    }

    reset = () => {
        this.sendReset();
        this.board = [
            [null, null, null],
            [null, null, null],
            [null, null, null],
        ];
        this.currentPlayer = 'X';
        this.winner = null;
        this.playable = true;
        this.actionPromise = null;
    }

    fromFlatIndex = (flatIndex) => {
        return {
            row: Math.floor(flatIndex / 3),
            col: flatIndex % 3,
        };
    }


    isBoardFull = () => {
        return !this.playable
    }

    isGameOver = () => {
        return this.winner || this.isBoardFull();
    }

    getWinner = () => {
        return this.winner;
    }

}

const board = document.querySelector('.board');
const cells = board.querySelectorAll('.cell');


var gameState = null;
var socket = null

var port = window.saucer.call('get_port', [])
console.log(port);
port.then((port) => {

    console.log(port);

    socket = new WebSocket(`ws://0.0.0.0:${port}`);
    gameState = new GameState(socket, board, cells);


    cells.forEach((cell, flatIndex) => {
        cell.addEventListener('click', async () => {
            const { row, col } = gameState.fromFlatIndex(flatIndex);

            if (gameState.isGameOver()) {
                return;
            }

            await gameState.sendMove(row, col);
            render();
        });
    });


    function render() {

        cells.forEach((cell, flatIndex) => {
            const { row, col } = gameState.fromFlatIndex(flatIndex);
            cell.textContent = gameState.board[row][col];
            cell.classList.toggle('cell-x', gameState.board[row][col] === 'X');
            cell.classList.toggle('cell-o', gameState.board[row][col] === 'O');
        });

        const winner = gameState.getWinner();
        if (winner) {
            alert(`Player ${winner} wins!`);
        } else if (gameState.isBoardFull()) {
            alert('It\'s a draw!');
        }
    }


    document.querySelector('.reset-button').addEventListener('click', () => {
        gameState.reset();
        render();
    });

});