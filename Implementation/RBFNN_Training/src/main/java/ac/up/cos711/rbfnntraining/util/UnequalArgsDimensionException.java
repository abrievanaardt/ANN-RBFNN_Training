package ac.up.cos711.rbfnntraining.util;

/**
 *
 * @author Abrie van Aardt
 */
public class UnequalArgsDimensionException extends Exception{
    public UnequalArgsDimensionException(String msg){
        super(msg);
    }
    
    public UnequalArgsDimensionException(){
        super();
    }    
    
}
