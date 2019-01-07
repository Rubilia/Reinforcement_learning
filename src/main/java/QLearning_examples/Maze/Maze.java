package QLearning_examples.Maze;

import DQN_learning.Environment.Environment;
import DQN_learning.Environment.State;
import DQN_learning.Environment.Step;
import DQN_learning.Learner.Learner;
import Tools.Pair;

import java.util.ArrayList;
import java.util.List;

public class Maze extends Environment {
    private double[][] maze;
    public double pathLenght = 0.0;
    private final double wallValue = -1.0, freeValue = 0.0, AgentValue = 10.0;
    private List<Pair<Integer, Integer>> Stack;
    private int h, w, limit, stepCounter;
    private double WallReward = -0.1, ReturnReward = -0.1, MoveRward = 0.2, EndReward = 30.0, LimReward, RewardCounter;
    public int Xcurrent = 1, Ycurrent = 1, Xend, Yend;
    public Maze(int h, int w, int limit) throws Exception {
        if (h*w%2==0){throw new Exception("Unable to create a maze with even side");}
        this.limit=limit;
        this.h = h;
        this.w = w;
        maze = new double[h][w];
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                if((i % 2 != 0  && j % 2 != 0) && (i < h-1 && j < w-1)) maze[i][j] = freeValue;
                else maze[i][j] = wallValue;
            }
        }
        MazeGenerator maze = new MazeGenerator(w/2, h/2);
        this.maze = convert(maze.maze, w/2, h/2);
        this.LimReward = -h*w/16.0;
        this.RewardCounter = 0.0;
        this.Stack = new ArrayList<>();
        this.stepCounter=0;
        Xend = w-2;
        Yend = h-2;
    }
    public double[][] getMaze(){return maze;}
    public double[][] convert(double[][] maze, int x, int y) {
        double[][] ret = new double[w][h];
        for (int i = 0; i < y; i++) {
            for (int j = 0; j < x; j++) {
                ret[2*j][2*i] = wallValue;
                ret[2*j+1][2*i] = ((int)maze[j][i] & 1) == 0 ? wallValue : freeValue;
            }
            ret[2*x][2*i] = wallValue;
            for (int j = 0; j < x; j++) {
                ret[2*j][2*i+1] = ((int)maze[j][i] & 8) == 0 ? wallValue: freeValue;
                ret[2*j+1][2*i+1] = freeValue;
            }
            ret[2*x][2*i+1] = wallValue;
        }
        for (int j = 0; j < x; j++) {
            ret[2*j][2*y] = wallValue;
            ret[2*j+1][2*y] = wallValue;
        }
        ret[2*x][2*y] = wallValue;
        return ret;
    }
    private State getInput(){
        double[][][][] inputConv = new double[1][1][w-2][h-2];
        double[] input = new double[(h-2)*(w-2)];
        for (int x = 1; x < w-1; x++) {
            for (int y = 1; y < h-1; y++) {
                inputConv[0][0][x-1][y-1] = (x==Xcurrent&&y==Ycurrent)?AgentValue:maze[x][y];
                input[x-1+(y-1)*(w-2)] = (x==Xcurrent&&y==Ycurrent)?AgentValue:maze[x][y];
            }
        }
        return new State(input, inputConv);
    }
    public void printMaze(){
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (maze[j][i]==wallValue) System.out.print("**");
                else System.out.print("  ");
            }
            System.out.print("\n");
        }
    }
    @Override
    public String toString() {
        String ret = "";
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (maze[j][i]==wallValue) ret+=("***");
                else if (j==Xcurrent&&i==Ycurrent)ret +=" @ ";
                else ret+=("   ");
            }
            ret+="\n";
        }
        return ret; }
    @Override
    public Step performAction(int a) {
        //Actions: 0 - up, 1 - down, 2 - left, 3 - right;
        State lastState = getInput(), newState;
        int xNew = Xcurrent, yNew = Ycurrent;
        if (a==0)yNew--;
        else if (a==1)yNew++;
        else if (a==2)xNew--;
        else xNew++;
        double R = MoveRward;
        if (maze[xNew][yNew]==wallValue){R=WallReward;xNew=Xcurrent;yNew=Ycurrent;}
        else if (xNew==Xend&&yNew==Yend){R=EndReward;}
        else { if (Stack.contains(new Pair<>(xNew, yNew)))R=ReturnReward;else{pathLenght+=1;}}
        RewardCounter+=R;
        Xcurrent = xNew;
        Ycurrent = yNew;
        newState = getInput();
        if (!Stack.contains(new Pair<>(Xcurrent, Ycurrent))){Stack.add(new Pair<>(Xcurrent, Ycurrent));}
        stepCounter++;
        return new Step(lastState, a, R, newState, isEnd());
    }
    @Override
    public boolean isEnd() {
        return (Xcurrent==Xend&&Ycurrent==Yend)||(RewardCounter<=LimReward)||(stepCounter>=limit);
    }
    @Override
    public void reset() {
        pathLenght=0;
        this.RewardCounter=0.0;
        Stack = new ArrayList<>();
        this.Xcurrent = 1;
        this.Ycurrent = 1;
        this.stepCounter=0;
    }

    @Override
    public State getCurrentState() {
        return getInput();
    }

    @Override
    public Pair<String, Boolean> getScore(Learner dqn){
        Maze m = null;
        try {
            m = (Maze) this.clone();
        } catch (Exception e) {
        }
        m.reset();
        while (!m.isEnd()) {
            m.performAction(dqn.produceActionGreedy(m.getCurrentState(), dqn.getTargetNetwork()));
        }
        return new Pair<>("avg path: " + m.pathLenght + "; " + ((m.Xcurrent == m.Xend || m.Ycurrent == m.Yend) ? "end reached" : "end isn`t reached"), (m.Xcurrent == m.Xend || m.Ycurrent == m.Yend));
    }
    @Override
    public Environment clone() {
        try {
            Maze m = new Maze(h, w, limit);
            m.maze=maze;
            m.pathLenght=pathLenght;
            m.Xcurrent=Xcurrent;
            m.stepCounter=stepCounter;
            m.Ycurrent=Ycurrent;
            m.Stack=Stack;
            m.RewardCounter=RewardCounter;
            return m;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}