/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ac.up.cos711.rbfnntraining.main;

import ac.up.cos711.rbfnntraining.data.util.StudyLogFormatter;
import ac.up.cos711.rbfnntraining.experiment.GDExperiment;
import ac.up.cos711.rbfnntraining.experiment.TwoPhaseExperiment;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.io.IOException;
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

        int simulations = 30;
        int epochs = 300;
        double acceptableError = 0.00000001;
        double w_l = 0.02;
        double u_l = 0.02;
        double sigma_l = 0.02;
        double dmax = 1;
        double rigor = 0.15;

        //this is an initial rate, it's decayed with epochs
        double lvq_l = 0.02;
        //neighbourhood radius, 0 is used
        int n_Radius = 0;
        try {
            setupLogging();
            for (int i = 2; i <= 15; i++) {
                /**
                 *
                 * @param _numSim
                 * @param _maxEpoch
                 * @param _numCentres
                 * @param _acceptableTError
                 * @param w_learningRate
                 * @param u_learningRate
                 * @param sigma_learningRate
                 * @param d_max
                 * @param _classificationRigor
                 */
//                new GDExperiment(simulations, epochs, i, acceptableError, w_l, u_l, sigma_l, dmax, rigor).start();
                new TwoPhaseExperiment(simulations, epochs, i, acceptableError, w_l, lvq_l, n_Radius, rigor).start();
            }

        }
        catch (Exception ex) {
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
