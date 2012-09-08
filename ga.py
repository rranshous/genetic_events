import random as r
from eventapp import EventApp
import json
from itertools import izip

# helper for those primitives
def _int(p):
    try:
        return int(str(p))
    except (AttributeError, TypeError):
        # protect against redis natives
        return 0

def _create_child(p1, p2):
    """
    creates a child by taking 1/2 the genes from each parent
    each gene is put up against random to choose parent
    """

    child = []
    # randomly choose from one of the parents genes
    for genes in izip(p1, p2):
        child.append(genes[r.randrange(0,2)])
    return child

def note_params(target_population_size, image_size, target_image,
                run_id, _string):
    # note our params, namespaced by the run_id
    run_id = str(run_id)
    target_population_size = _string('target_population_size:'+run_id,
                                  target_population_size)
    assert int(str(target_population_size)) >= 2, \
            "Must have a population 2 or more"
    image_size_x = _string('image_size:x:'+run_id, image_size[0])
    image_size_y = _string('image_size:y:'+run_id, image_size[1])
    target_image = _string('target_image:'+run_id, target_image)

    # change event name and keep data
    yield ( 'run_id', run_id )


def create_initial_population(target_population_size, image_size,
                              run_id):

    # fill the population w/ chromosomes
    for i in xrange(target_population_size):

        # create a gene for each pixel of the target image
        chromosome = []
        for j in xrange(image_size[0] * image_size[1]):

            # create a random gene with a random x and y within
            # the image bounds, and a random value
            gene = [r.randrange(0, image_size[0]),
                    r.randrange(0, image_size[1]),
                    r.sample([0,1], 1)[0]]

            # add the gene to our chromosome
            chromosome.append(gene)

        # our resulting event will have our chromosome definition
        # and the run it belongs to
        yield dict( chromosome = chromosome,
                    run_id = run_id )


def calculate_cost(chromosome, run_id, _string):
    """
    compare's the chromosome against the target image, calculates
    the cost based on how different they are
    """

    # get the image bounds
    image_x = _int(_string('image_size:x:'+run_id))
    image_y = _int(_string('image_size:y:'+run_id))
    target_image = _string('target_image:'+run_id)

    cost = 0
    chromosome_image = [' ' for x in xrange(image_x * image_y)]

    # go through the gene's creating our chromosome image
    for gene in chromosome:
        try:
            gene_x, gene_y, gene_value = gene
        except Exception, ex:
            raise Exception( 'gene exception: %s %s' % (type(chromosome), str(chromosome)))

        offset = gene_x + gene_y * image_x
        chromosome_image[offset] = gene_value

    # now compare our resulting image to the target image
    for i in xrange(image_x * image_y):
        if int(target_image[i]) != chromosome_image[i]:
            cost += 1

    # now we know the cost, yield up an event announcing we found it
    yield dict( chromosome = chromosome,
                run_id = run_id,
                cost = cost )


def filter_costly_chromosomes(chromosome, run_id, cost, _sequence, _string,
                              _event, event_data):
    """
    will filter all chromosome's which have an above avg cost

    bases this avg on a sample of the population
    """

    # get our sample of the popuation
    sample_ratio = .1
    target_population_size = _int(_string('target_population_size:'+run_id))
    population_size = _int(_string('population_size:'+run_id))
    sample_size = min(min(population_size, target_population_size) * sample_ratio, 3)
    sample = _sequence('costly_filter_sample:'+run_id)
    print 'costly sample_size: %s' % sample_size

    # don't wanna divide by 0
    if len(sample) == 0:
        sample_avg_cost = False
    else:
        sample_avg_cost = sum(map(int,sample)) / len(sample)

    # add our chromosome's cost to the sample, add it to the
    # left side of the list, this way we can pop off the right
    sample.push_head(cost)

    print 'costly sample len: %s' % len(sample)

    # if the sample is too small, than we don't have enough
    # data to make a determination so we'll just let it go on
    if len(sample) < sample_size:
        print 'costly sample too small'
        yield True

    # trim the sample to size
    else:
        # we added one, lets remove one
        sample.pop_tail()

    # if the chromosome is not too costly, than pass it on
    print 'cost: %s :: sample_avg_cost: %s' % (cost, sample_avg_cost)
    if sample_avg_cost is False or cost <= sample_avg_cost:
        yield True

    else:
        yield _event('found_above_cost_chromosome', event_data)


def mate_chromosomes(chromosome, run_id, _sequence, _string):
    """
    mates chromosomes from the same run, produces two chromosomes from
    two chromosomes

    will gather up chromosomes from a run until we have a very small sample
    of the population size, than mates two from that sample

    this means that the first few events we get we will not act on
    """

    # get our datas
    sample_ratio = .1
    target_population_size = _int(_string('target_population_size:'+run_id))
    population_size = _int(_string('population_size:'+run_id))
    sample = _sequence('mate_chromosomes_sample:'+run_id)

    # we want to try and keep the population size
    # about the same, so we need to have those who mate
    # mate a lot if not that many people are mating
    # we'll have enough children as to reduce the difference between
    # our current population and the target population by 1/10
    print 'mate target_population_size: %s' % target_population_size
    print 'mate population_size: %s' % population_size
    print 'mate target_population diff: %s' % (target_population_size - population_size)
    number_of_children = int((target_population_size - population_size) * .1)
    # we want at least 1 child, we're that awesome
    number_of_children = max(1, number_of_children)
    print 'mate number of children: %s' % number_of_children

    # add our chromosome to the sample
    sample.push_tail(json.dumps(chromosome))

    # our sample size has to be at least two large
    # (it takes two to tango)
    min_sample_size = max(min(target_population_size, population_size) * sample_ratio, 2)
    print 'mate min_sample_size: %s' % min_sample_size
    print 'mate sample len: %s' % (len(sample))

    # if the sample is large enough, pick two chromosome's to mate
    if len(sample) >= min_sample_size:

        # pop two chromosomes from the sample
        chrom1, chrom2 = sample.pop_head(), sample.pop_head()

        # if we only got one person from the sample, add
        # the other one back in (lost opertunity)
        if not chrom2:
            sample.push_head(chrom1)

            # and call the whole thing off
            yield False

        print 'mating'

        # de-serialize
        chrom1, chrom2 = map(json.loads, (chrom1, chrom2))

        # split the chromosomes and mate them
        for i in xrange(number_of_children):
            child = _create_child(chrom1, chrom2)

            # yield up an event for each new chromosome
            yield dict( chromosome = child,
                        parents = [chrom1, chrom2],
                        run_id = run_id )

        print 'mate post sample len: %s' % len(sample)

    else:
        print 'mate sample too small'



def mutate_chromosome(chromosome, run_id, event, _string):
    """
    mutates the given chromosome
    """

    # mutate 10 percent of the population
    mutation_rate = 0.5
    mutation_severity = 0.1

    # get our image constraints
    image_size_x = _int(_string('image_size:x:'+run_id))
    image_size_y = _int(_string('image_size:y:'+run_id))
    image_size = [image_size_x, image_size_y]

    # are we lucky enough to mutate ?
    if r.random() <= mutation_rate:

        # keep a copy for later
        original_chromosome = chromosome[:]

        # we mutate the chromosome by changing one of it's
        # gene's, it's x, its y, or it's value
        choice = r.sample([0,1,2], 1)[0]
        gene = r.sample(chromosome, 1)[0]

        # change x or y
        if choice in (0, 1):
            # change it's X limiting to possible range, at least 1
            skew = max(int(mutation_severity * image_size[choice]), 1)
            # limit to 0 -> image size
            top = min(image_size[choice], gene[choice] + skew)
            bottom = max(0, gene[choice] - skew)
            try:
                gene[choice] = r.randrange(bottom, top)
            except Exception:
                raise Exception('choice: %s\ngenechoice: %s\nbottom: %s\ntop: %s\nskew: %s\nimage size: %s'
                                % (choice, gene[choice], bottom, top, skew, str(image_size)))

        # change value
        if choice == 2:
            # since the value is 0 or 1, we're going to only change
            # to the other if the mutation severity pans out
            if r.random() <= mutation_severity:
                gene[2] = 0 if gene[2] is 1 else 1

        yield dict( chromosome = chromosome,
                    original_chromosome = original_chromosome,
                    run_id = run_id )


def reintroduce_mutant(chromosome, run_id):
    yield True

def increment_population(_string, run_id):
    population = _string('population_size:'+run_id)
    population.incr()
    print '[I] population_size %s' % str(population)
    yield False

def decrement_population(_string, run_id):
    population = _string('population_size:'+run_id)
    population.decr()
    print '[D] population_size %s' % str(population)
    yield False


# NOTE: It feels like there is somethign wrong w/ this flow
app  = EventApp('ga_pic',

                # save to redis our run's params
                ('start_pic_ga', note_params, 'pic_ga_started'),

                # create our initial population
                (create_initial_population, 'created_chromosome'),

                # calculate the cost of all new chromosomes
                ('created_chromosome', calculate_cost, 'calculated_chromosome_cost'),

                # filter above avg cost chromosomes
                (filter_costly_chromosomes, 'found_below_cost_chromosome'),

                # mate the good chromosomes, only good looking ppl get laid
                (mate_chromosomes, 'created_chromosome'),

                # mutate chromosomes, this will also mutate the initial population
                ('created_chromosome', mutate_chromosome, 'mutated_chromosome'),

                # once we've mutated a chromosome we want to re-introduce it
                # to the pool as a created chromosome
                # NOTE: this works around each handler only being able to put
                #       off a single event type easily
                (reintroduce_mutant, 'created_chromosome'),

                # we want to keep a estimate of the total population
                ('created_chromosome', increment_population, '_'),
                ('found_above_cost_chromosome', decrement_population, '_')

)
app.run()
