import random as r
from eventapp import EventApp
import json

# helper for those primitives
def _int(p):
    return int(str(p))

def note_params(target_population_size, image_size, target_image,
                run_id, _string):
    # note our params, namespaced by the run_id
    run_id = str(run_id)
    target_population_size = _string('target_population_size:'+run_id,
                                  target_population_size)
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


def filter_costly_chromosomes(chromosome, run_id, cost, _sequence, _string):
    """
    will filter all chromosome's which have an above avg cost

    bases this avg on a sample of the population
    """

    # get our sample of the popuation
    sample_ratio = .5
    target_population_size = _int(_string('target_population_size:'+run_id))
    sample_size = target_population_size * sample_ratio
    sample = _sequence('costly_filter_sample:'+run_id)

    # don't wanna divide by 0
    if len(sample) == 0:
        sample_avg_cost = False
    else:
        sample_avg_cost = sum(map(int,sample)) / len(sample)

    # add our chromosome's cost to the sample, add it to the
    # left side of the list, this way we can pop off the right
    sample.push_head(cost)

    # if the sample is too small, than we don't have enough
    # data to make a determination so we'll just let it go on
    if len(sample) < sample_size:
        yield True

    # trim the sample to size
    if len(sample) > sample_size:
        # we added one, lets remove one
        sample.pop_tail()

    # if the chromosome is not too costly, than pass it on
    # True = Keep
    print 'cost: %s :: sample_avg_cost: %s' % (cost, sample_avg_cost)
    if sample_avg_cost is False or cost <= sample_avg_cost:
        yield True
    else:
        yield False


def mate_chromosomes(chromosome, run_id, _sequence, _string):
    """
    mates chromosomes from the same run, produces two chromosomes from
    two chromosomes

    will gather up chromosomes from a run until we have a very small sample
    of the population size, than mates two from that sample

    this means that the first few events we get we will not act on
    """

    # get our sample of the population
    sample_ratio = .02
    target_population_size = _int(_string('target_population_size:'+run_id))
    # must have at least two to tango
    target_population_size = max(2, target_population_size)
    # our sample size has to be at least two large
    sample_size = max(target_population_size * sample_ratio, 2)
    sample = _sequence('mate_chromosomes_sample:'+run_id)

    print 'mate sample: %s' % [x for x in sample]

    # add our chromosome to the sample at a random location
    if len(sample) in (0, 1):
        # no point in being random if there are no other values
        # encode to store in redis
        sample.push_tail(json.dumps(chromosome))
    else:
        index = r.randrange(o, len(sample)-1)
        # NOTE: would use insert here but redis natives replaces
        #       not adds on insert
        sample = _sequence('mate_chromomsomes_sample:'+run_id,
                           sample[:index] +
                            [json.dumps(chromosome)] +
                            sample[index:])

    print 'mate post add sample: %s' % [x for x in sample]

    # if the sample is large enough, pick two chromosome's to mate
    if len(sample) >= sample_size:

        print 'mating'

        # pop two chromosomes from the sample
        chrom1, chrom2 = sample.pop_head(), sample.pop_head()
        # de-serialize
        chrom1, chrom2 = map(json.loads, (chrom1, chrom2))

        # split the chromosomes and mate them
        mid = len(chrom1) / 2
        chrom1_new = chrom1[0:mid] + chrom2[mid:]
        chrom2_new = chrom2[0:mid] + chrom1[mid:]

        # yield up an event for each new chromosome
        yield dict( chromosome = chrom1_new,
                    parents = [chrom1, chrom2],
                    run_id = run_id )

        yield dict( chromosome = chrom2_new,
                    parents = [chrom1, chrom2],
                    run_id = run_id )



def mutate_chromosome(chromosome, run_id, event, _string):
    """
    mutates the given chromosome
    """

    # mutate 10 percent of the population
    mutation_rate = 0.1
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

# NOTE: It feels like there is somethign wrong w/ this flow
app  = EventApp('ga_pic',

                # save to redis our run's params
                ('start_pic_ga', note_params, 'pic_ga_started'),

                # create our initial population
                (create_initial_population, 'created_chromosome'),

                # calculate the cost of all new chromosomes
                (calculate_cost, 'calculated_chromosome_cost'),

                # filter above avg cost chromosomes
                (filter_costly_chromosomes, 'found_below_cost_chromosome'),

                # mate the good chromosomes, the result is a created chromosome
                (mate_chromosomes, 'created_chromosome'),

                # mutate chromosomes, this will also mutate the initial population
                # NOTE: could this be feating off created chromosome?
                (mutate_chromosome, 'mutated_chromosome'),

                # once we've mutated a chromosome we want to re-introduce it
                # to the pool as a created chromosome
                # NOTE: this works around each handler only being able to put
                #       off a single event type easily
                (reintroduce_mutant, 'created_chromosome')

)
app.run()
