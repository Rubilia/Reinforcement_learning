package Tools;

public class Pair<K, V>{
    private K key;
    private V value;
    public Pair(K key, V value){
        this.key = key;
        this.value = value;
    }
    public K getKey(){
        return key;
    }
    public V getValue(){
        return value;
    }
    public void setKey(K key) {
        this.key = key;
    }
    public void setValue(V value) {
        this.value = value;
    }
    @Override
    public String toString() {
        return key.toString()+": " + value.toString();
    }

    @Override
    public boolean equals(Object obj) {
        Pair<K, V> pair = (Pair<K, V>)obj;
        boolean b1 = this.key.equals(pair.key), b2 = this.value.equals(pair.value);
        return b1&&b2;
    }
}
