from fitness.base_ff_classes.base_ff import base_ff
from utilities.fitness.get_data import get_data
from algorithm.parameters import params
from algorithm.mapper import map_tree_from_genome
from representation.individual import Individual 

class renewable_energy(base_ff):
    """
    Basic fitness function template for writing new fitness functions. This
    basic template inherits from the base fitness function class, which
    contains various checks and balances.
    
    Note that all fitness functions must be implemented as a class.
    
    Note that the class name must be the same as the file name.
    
    Important points to note about base fitness function class from which
    this template inherits:
    
      - Default Fitness values (can be referenced as "self.default_fitness")
        are set to NaN in the base class. While this can be over-written,
        PonyGE2 works best when it can filter solutions by NaN values.
    
      - The standard fitness objective of the base fitness function class is
        to minimise fitness. If the objective is to maximise fitness,
        this can be over-written by setting the flag "maximise = True".
    
    """

    # The base fitness function class is set up to minimise fitness.
    # However, if you wish to maximise fitness values, you only need to
    # change the "maximise" attribute here to True rather than False.
    # Note that if fitness is being minimised, it is not necessary to
    # re-define/overwrite the maximise attribute here, as it already exists
    # in the base fitness function class.
    maximise = False

    def __init__(self):
        """
        All fitness functions which inherit from the bass fitness function
        class must initialise the base class during their own initialisation.
        """

        # Initialise base fitness function class.
        super().__init__()
        # Get training and test data
        self.X_train, self.y_train, self.X_test, self.y_test = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])
        
    # def parse_phenotype(self, ind):
    #     """
    #     Parse the phenotype of the individual to extract the model parameters
    #     and structure.
        
    #     :param ind: An individual whose phenotype is to be parsed.
    #     :return: Parsed model parameters and structure.
    #     """
    #     # This is a placeholder for parsing logic, which should be implemented
    #     # based on the specific requirements of the renewable energy problem.
    #     # For example, it could extract hyperparameters for an LSTM or GRU model.

        
    #     ind = Individual(ind.genome, ind.tree, map_ind=True)
    #     return ind.phenotype
    


    def evaluate(self, ind, **kwargs):
        """
        Default fitness execution call for all fitness functions. When
        implementing a new fitness function, this is where code should be added
        to evaluate target phenotypes.
        
        There is no need to implement a __call__() method for new fitness
        functions which inherit from the base class; the "evaluate()" function
        provided here allows for this. Implementing a __call__() method for new
        fitness functions will over-write the __call__() method in the base
        class, removing much of the functionality and use of the base class.
                
        :param ind: An individual to be evaluated.
        :param kwargs: Optional extra arguments.
        :return: The fitness of the evaluated individual.
        """
        """
        Avalia o indivíduo com base no RMSE do modelo gerado a partir do fenótipo.
        """

        ind = Individual(None, ind.tree, map_ind=True)
        # Ensure the phenotype is correctly parsed
        if ind.phenotype is None:
            raise ValueError("O fenótipo do indivíduo não pôde ser analisado.")
        # Here we assume the phenotype is a valid Python expression that can be evaluated
        print("[DEBUG] Phenotype:", ind.phenotype)
        # Evaluate the fitness of the phenotype
        fitness = eval(ind.phenotype)

        return fitness