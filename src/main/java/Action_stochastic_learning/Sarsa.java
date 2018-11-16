package Action_stochastic_learning;

import Tools.Graph;
import Tools.Pair;

import java.util.*;

public class Sarsa {
    static Value_Storage<Pair<Integer, Integer>, double[]> Q;
    static List<Integer> graphIndexes = new ArrayList<>();
    final static Pair<Integer, Integer>[] negativePoints = new Pair[] {new Pair<>(5, 5), new Pair<>(6, 5), new Pair<>(3, 4)};
    final static Pair<Integer, Integer> positivePoint = new Pair<>(5, 6), finalState = new Pair<>(6, 6);
    static double epsilon = 0.5, alpha, y = 0.8;
    static int ActionSpaceSize = 4;
    static int epoch = 0, epochsAmount = 1, iteration = 0, iterationsAmount = 200;
    public static void main(String[] args) {
        graphIndexes.add(0);
        alpha = 5/(double)iterationsAmount;
        Q = initQ();
        for (epoch = 0; epoch < epochsAmount; epoch++){
            evaluate.run();
            iteration = 0;
        }
    }
    static Value_Storage<Pair<Integer, Integer>, double[]> initQ(){
        Value_Storage<Pair<Integer, Integer>, double[]> ret;
        ret = new Value_Storage<>();
        for (int x = 0; x < 7; x++) {
            for (int y = 0; y < 7; y++) {
                ret.add(new Pair<>(x, y), new double[ActionSpaceSize]);
            }
        }
        return ret;
    }
    static Runnable evaluate = new Runnable() {
        //0-up, 1-down, 2-left, 3-right
        Pair<Integer, Integer> currentState, lastState;
        Random rnd = new Random();
        double stepcounter;
        public void init(){
            stepcounter = 0.0;
            currentState = new Pair<>(rnd.nextInt(2), rnd.nextInt(2));
            lastState = currentState;
        }
        @Override
        public void run() {
            init();
            if (epoch==0&&graphIndexes.contains(epoch)){ Graph.buildPolicy2D(visualizePolicy(), 7, "Agents policy before training "); }
            while (iteration<iterationsAmount){
                boolean b = true, c = true;
                while (c){
                    int A = produceAction(false, currentState), A_new;
                    double R = executeAction(A);
                    double[] Q_values = Q.get(lastState);
                    A_new = produceAction(true, currentState);
                    Q_values[A] = Q_values[A] + alpha*(R + y*Q.get(currentState)[A_new] - Q_values[A]);
                    Q.set(lastState, Q_values);
                    if (!b){c = false;}
                    if (currentState.equals(finalState)){b = false;}
                    stepcounter++;
                }
                iteration++;
                init();
            }
            if (graphIndexes.contains(epoch)){ Graph.buildPolicy2D(visualizePolicy(), 7, "Agents policy after " + (epoch+1) + " epochs"); }
        }
        int produceAction(boolean greedy, Pair<Integer, Integer> state){
            if (rnd.nextDouble()<epsilon&&!greedy){return rnd.nextInt(ActionSpaceSize);}
            List<Integer> maxs = argsMax(Q.get(state));
            return maxs.get(rnd.nextInt(maxs.size()));
        }
        double executeAction(int action){
            Pair<Integer, Integer> next = currentState;
            double reward = 0.0;
            if (action==0){
                if (currentState.getValue()!=6) next = new Pair<>(currentState.getKey(), currentState.getValue()+1);
            }
            else if (action==1){
                if (currentState.getValue()!=0){next = new Pair<>(currentState.getKey(), currentState.getValue()-1);}
            }
            else if (action==2){
                if (currentState.getKey()!=0){next=new Pair<>(currentState.getKey()-1, currentState.getValue());}
            }
            else{
                if (currentState.getKey()!=6){next = new Pair<>(currentState.getKey()+1, currentState.getValue());}
            }
            if (isInsideWall(next)){next=currentState;}
            if (contains(negativePoints, next)){reward+=-1.8;}
            else if (next.equals(positivePoint)){reward+=0.05;}
            else if (next.equals(finalState))
            { reward = 10*12/(stepcounter); }
            lastState = currentState;
            currentState = next;
            return reward;
        }
        boolean isInsideWall(Pair<Integer, Integer> s){
            return (s.getKey()>=1&&s.getKey()<=2&&s.getValue()>=4)||(s.getKey()>=3&&s.getKey()<=4&&s.getValue()<=2);
        }
        List<Integer> argsMax(double[] array){
            List<Integer> ret = new ArrayList<>();
            double maxValue = Max(array);
            for (int i = 0; i < array.length; i++) {
                if (array[i]==maxValue){ret.add(i);}
            }
            return ret;
        }
        double Max(double[] inp){
            int i = 0;
            for (int j = 1; j < inp.length; j++) {
                if (inp[j]>inp[i]){i=j;}
            }
            return inp[i];
        }
        String[][] visualizePolicy(){
            String[][] ret = new String[7][7];
            for (int y = 6; y >= 0; y--) {
                for (int x = 0; x <7; x++) {
                    if (isInsideWall(new Pair<>(x, y))||(x==6&&y==6)){ret[6-y][x] = " ";continue;}
                    int action = produceAction(true, new Pair<>(x, y));
                    if (action==0){ret[6-y][x] = "↑";}
                    else if (action==1){ret[6-y][x] = "↓";}
                    else if (action==2){ret[6-y][x] = "←";}
                    else if (action==3){ret[6-y][x] = "→";}
                }
            }
            return ret;
        }
    };
    static boolean contains(Pair<Integer, Integer>[] array, Pair<Integer, Integer> item){
        boolean ret = false;
        for(Pair<Integer, Integer> s : array){ret |= s.equals(item);}
        return ret;
    }
}
class Value_Storage<K, V>{
    List<K> keys;
    List<V> values;
    public Value_Storage(){
        keys = new ArrayList<>();
        values = new ArrayList<>();
    }
    public V get(K key){
        int index = IndexOf(key);
        if (index==-1){return null;}
        return values.get(index);
    }
    private int IndexOf(K key){
        for (int j = 0; j < keys.size(); j++) {
            if (key.equals(keys.get(j))){return j;}
        }
        return -1;
    }
    public void add(K key, V value){
        if (keys.contains(key)){this.set(key, value);return;}
        keys.add(key);
        values.add(value);
    }
    public void set(K key, V value){
        if (!keys.contains(key)){return;}
        values.set(keys.indexOf(key), value);
    }
}