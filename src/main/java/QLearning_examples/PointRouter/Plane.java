package QLearning_examples.PointRouter;

import DQN_learning.Environment.Environment;
import DQN_learning.Environment.State;
import DQN_learning.Environment.Step;
import DQN_learning.Learner.Learner;
import Tools.OtherTools;
import Tools.Pair;

import java.util.Random;

public class Plane extends Environment {
    protected int x, y, Fx, Fy, n, limit, counter;
    private final double EmptyValue = 0.0, AgentValue = 1.0, PointValue = -1.0;
    private final double NothingReward = 0.0, ReachedReward = 1.0, wrongReward = -0.1;
    Random rnd = new Random();
    public Plane(int n, int limit){
        this.n = n;
        this.limit = limit;
        x = rnd.nextInt(n);
        y = rnd.nextInt(n);
        do {
            Fx = rnd.nextInt(n);
        }while (Fx==x);
        do {
            Fy = rnd.nextInt(n);
        }while (Fy==y);
        counter = 0;
    }
    @Override
    public Step performAction(int a) {
        State begin = getCurrentState();
        double reward = NothingReward;
        if (a==0) y++;
        else if (a==1)y--;
        else if (a==2)x++;
        else x--;
        if (y < 0){y=0;reward = wrongReward;}
        if (x < 0){ x=0;reward = wrongReward;}
        if (x>=n){x = n-1; reward = wrongReward;}
        if (y>=n){y = n-1; reward = wrongReward;}
        if (x==Fx&&y==Fy){
            reward=ReachedReward;
            do {
                Fx = rnd.nextInt(n);
            }while (Fx==x);
            do {
                Fy = rnd.nextInt(n);
            }while (Fy==y);
        }
        counter++;
        return new Step(begin, a, reward, getCurrentState(), isEnd());
    }

    @Override
    public boolean isEnd() {
        return counter>=limit;
    }

    @Override
    public void reset() {
        x = rnd.nextInt(n);
        y = rnd.nextInt(n);
        do {
            Fx = rnd.nextInt(n);
        }while (Fx==x);
        do {
            Fy = rnd.nextInt(n);
        }while (Fy==y);
        counter = 0;
    }
    private double[][] getPlane(){
//        double[][] plane = new double[n][n];
//        for (int x = n-1; x >= 0; x--) {
//            for (int y = 0; y < n; y++) {
//                plane[x][y] = (x==this.x&&y==this.y)?AgentValue:(x==Fx&&y==Fy)?PointValue:EmptyValue;
//            }
//        }
        double[][] plane = new double[][]{{x,y}, {Fx, Fy}};
        return plane;
    }
    @Override
    public State getCurrentState() {
        double[][] tmpValue = getPlane();
        return new State(OtherTools.convert2to1(tmpValue), new double[][][][]{{tmpValue}});
    }

    @Override
    public Pair<String, Boolean> getScore(Learner dqn) {
        double n = this.n*10, score = 0.0;
        for (int i = 0; i < n; i++) {
            score+=score(dqn);
        }
        score/=n;
        return new Pair<>("got score " + score + ", max score:" + Math.max(limit/(2*this.n-1), score), score>=(limit/(2*this.n-1)));
    }
    private double score(Learner dqn){
        Environment env = this.clone();
        int pointCounter = 0;
        Step s;
        while (!env.isEnd()){
            s = env.performAction(dqn.produceActionGreedy(env.getCurrentState(), dqn.getTargetNetwork()));
            if (s.getR()==ReachedReward)pointCounter++;
        }
        return (double)pointCounter;
    }

    @Override
    public Environment clone() {
        return new Plane(n, limit);
    }
}
