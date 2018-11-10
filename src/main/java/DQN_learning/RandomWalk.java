package DQN_learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class RandomWalk extends Environment {
    private int n, currentPos, limit, counter;
    private boolean end = false;
    private final double leftReward = -1.0, rightReward = 1.0, otherReward = 0.0;
    private Random rnd = new Random();
    public RandomWalk(int n, int lim){
        this.n = n;
        this.limit = lim;
        this.counter = 0;
        this.currentPos = 1+ rnd.nextInt(n-2);
    }
    @Override
    public Step performAction(int a) {
        double reward = otherReward;
        boolean b = true;
        INDArray S = Nd4j.create(new double[]{(double)currentPos/(double)n}), newS;
        if (a==0){ currentPos--; }
        else{currentPos++;}
        if (currentPos==-1){reward = leftReward;newS = Nd4j.create(new double[]{-1.0});end=true;}
        else if (currentPos==n){reward=rightReward;newS = Nd4j.create(new double[]{-0.5});end=true;}
        else{newS = Nd4j.create(new double[]{(double)currentPos/(double)n});b=false;}
        counter++;
        return  new Step(new State(S), a, reward, new State(newS), b);
    }

    @Override
    public boolean isEnd() {
        return !(!end&&(counter<limit||limit==-1));
    }

    @Override
    public void reset() {
        this.counter = 0;
        this.currentPos = 1+ rnd.nextInt(n-2);
        this.end = false;
    }

    @Override
    public State getCurrentState() {
        if (currentPos==-1){return new State(Nd4j.create(new double[]{-1.0}));}
        else if (currentPos==n){return new State(Nd4j.create(new double[]{-0.5}));}
        else{ return new State(Nd4j.create(new double[]{(double)currentPos/(double)n})); }
    }
}
