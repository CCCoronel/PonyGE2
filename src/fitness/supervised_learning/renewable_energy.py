from fitness.base_ff_classes.base_ff import base_ff
from utilities.fitness.get_data import get_data
from algorithm.parameters import params
from representation.individual import Individual
import re
import numpy as np

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
        phenotype = ind.phenotype
        
        try:
            
            # Create a function to fix numbers in the phenotype string
            def clean_phenotype_string(p_str):
                """
                Cleans a phenotype string generated by PonyGE2 to make it valid Python code.
                Fixes spaces within numbers, around decimal points, and inside string values.
                """
                p_str = re.sub(r'\s*\.\s*', '.', p_str)
                while re.search(r'(\d+)\s+(\d+)', p_str):
                    p_str = re.sub(r'(\d+)\s+(\d+)', r'\1\2', p_str)
                def strip_string_literal(match):
                    return f"'{match.group(1).strip()}'"
                p_str = re.sub(r"'([^']*)'", strip_string_literal, p_str)
                return p_str

            # Clean the phenotype string
            clean_phenotype = clean_phenotype_string(phenotype)
            
            
            # Use the cleaned phenotype to create a model
            model = eval(clean_phenotype)
            
            # ... resto da sua lógica de fitness ...

        except Exception as e:
            # Handle any exceptions that occur during evaluation
            # This could be due to syntax errors, runtime errors, etc.
            print(f"Erro ao avaliar fenótipo.\nOriginal: {phenotype}\nLimpo: {clean_phenotype}\nErro: {e}")
            fitness = 0.0
            
        return fitness