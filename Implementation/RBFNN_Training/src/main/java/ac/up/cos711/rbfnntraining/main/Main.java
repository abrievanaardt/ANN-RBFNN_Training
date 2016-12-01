/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ac.up.cos711.rbfnntraining.main;

import ac.up.cos711.rbfnntraining.data.Dataset;
import ac.up.cos711.rbfnntraining.data.Graph;
import ac.up.cos711.rbfnntraining.data.util.GraphException;
import ac.up.cos711.rbfnntraining.data.util.IncorrectFileFormatException;
import ac.up.cos711.rbfnntraining.data.util.StudyLogFormatter;
import ac.up.cos711.rbfnntraining.function.Identity;
import ac.up.cos711.rbfnntraining.function.util.NotAFunctionException;
import ac.up.cos711.rbfnntraining.neuralnet.util.FFNeuralNetBuilder;
import ac.up.cos711.rbfnntraining.neuralnet.util.ZeroNeuronException;
import java.io.FileNotFoundException;
import java.util.logging.Level;
import java.util.logging.Logger;
import ac.up.cos711.rbfnntraining.neuralnet.IRBFNeuralNet;
import ac.up.cos711.rbfnntraining.neuralnet.RBFNeuralNetTest;
import ac.up.cos711.rbfnntraining.neuralnet.training.GradientDescent;
import ac.up.cos711.rbfnntraining.neuralnet.util.ThresholdOutOfBoundsException;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;

/**
 *
 * @author Abrie
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String PATH_PREFIX = "ac/up/cos711/rbfnntraining/data/";
        String EXT = ".nsds";

        try {
            setupLogging();
            String[] datasetNames = new String[]{
//                "cancer", 
                "diabetes",
                            "glass",
                            "heart",
//                            "iris"
            };

            int numCentres = 15;
            double d_max = 1;
            int maxEpochs = 1000;

            for (int i = 0; i < datasetNames.length; i++) {
                Dataset dataset = Dataset
                        .fromFile(PATH_PREFIX
                                + "/" + datasetNames[i]
                                + EXT);
                dataset.setDatasetName(datasetNames[i]);
                List<Dataset> datasets = dataset.split(0.6, 0.3, 0.1);
                RBFNeuralNetTest network = new RBFNeuralNetTest(dataset.getInputCount(), numCentres, dataset.getTargetCount());
                GradientDescent gd = new GradientDescent(0.00000001, 0.02, 0.02, 0.02, d_max, 0.2, maxEpochs);

                gd.train(network, datasets.get(0), datasets.get(1));

               Graph graph = new Graph("Plots", "Test", "Epoch", "TrainingError", "", 2);
               graph.addPlot("Training", gd.getEpochs(), gd.getTrainingErrorHistory(), "line");
               graph.addPlot("Generalise", gd.getEpochs(), gd.getValidationErrorHistory(), "line");
               graph.plot();
            }
        }
        catch (FileNotFoundException | IncorrectFileFormatException | ThresholdOutOfBoundsException | UnequalArgsDimensionException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (GraphException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void setupLogging() throws IOException {
        Formatter logFormatter = new StudyLogFormatter();
        Logger.getLogger(Main.class.getName()).setLevel(Level.CONFIG);
        Logger.getLogger(ExecutorService.class.getName()).setLevel(Level.OFF);
        Logger logger = Logger.getLogger("");
        FileHandler logFileHandler = new FileHandler("log\\study.log", true);
        logFileHandler.setFormatter(logFormatter);
        logger.addHandler(logFileHandler);
        logger.setLevel(Level.ALL);
        logger.getHandlers()[0].setFormatter(logFormatter);
        logger.getHandlers()[0].setLevel(Level.ALL);//console output
        logger.getHandlers()[1].setLevel(Level.CONFIG);//normal log file
    }

}
