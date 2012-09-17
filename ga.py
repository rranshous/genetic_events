import random as r
import eventapp
from eventapp import EventApp
import json
from itertools import izip

eventapp.threads_per_stage = 4

# helper for those primitives
def _int(p):
    try:
        return int(str(p))
    except (AttributeError, TypeError):
        # protect against redis natives
        return 0

def _create_new_child(chromosome_length):
    """
    creates a new random chromosome
    """

    return [r.randint(0,1) for i in xrange(chromosome_length)]

def _create_child(p1=None, p2=None):
    """
    creates a child by folding the genes at a random location
    """

    index = r.randint(0, len(p1))
    return p1[index:] + p2[:index]

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
    lowest_cost = _int(_string('lowest_cost:'+run_id, 9999))

    # change event name and keep data
    yield ( 'run_id', run_id )


def create_initial_population(target_population_size, image_size,
                              run_id):

    chrom_len = image_size[0] * image_size[1]

    # fill the population w/ chromosomes
    for i in xrange(target_population_size):

        yield dict( chromosome = _create_new_child(chrom_len),
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

    # compare our chromosome against the target's
    for i in xrange(image_x * image_y):
        if int(target_image[i]) != chromosome[i]:
            cost += 1

    # now we know the cost, yield up an event announcing we found it
    yield ('cost', cost)


def filter_costly_chromosomes(chromosome, run_id, cost, _sequence, _string,
                              _event, _zset, event_data):
    """
    will filter all chromosome's which have an above avg cost
    """


    # pull our shared lowest cost counter
    lowest_cost = _string('lowest_cost:'+run_id)
    best_solutions = _zset('best_solutions:'+run_id)

    # to qualify as a low cost chromosome you
    # must be within 20% of the closest
    diff = cost - _int(lowest_cost)
    diff_percent = diff / float(cost)

    print '[FCC] cost: %s' % cost
    print '[FCC] lowest_cost: %s' % lowest_cost
    print '[FCC] diff: %s' % diff
    print '[FCC] diff_percent: %s' % diff_percent

    if diff_percent <= 0 or diff_percent < .1:
        print '[FCC] low cost: %s' % cost

        # add the new found solution and remove worst
        best_solutions.add(json.dumps(chromosome), cost)
        #if len(best_solutions) > 10:
        #    best_solutions.remove_range_by_rank(-2, -1)

        # we've found the new low !
        if cost < _int(lowest_cost):
            print '[FCC] new low: %s' % cost
            lowest_cost.value = cost

        yield ( 'cost_diff_percent', diff_percent )

    else:
        event_data['cost_diff_percent'] = diff_percent
        yield _event('found_above_cost_chromosome', event_data)


def mate_chromosomes(chromosome, cost_diff_percent, run_id, _sequence, _string):
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
    # our current population and the target population by 10%
    number_of_children = int((target_population_size - population_size) * .1)

    # we want at least 1 child, we're that awesome
    number_of_children = max(1, number_of_children)

    # now scale this by how good the cost is compared to the lowest
    number_of_children += int(-cost_diff_percent * number_of_children * 10)

    # we want at least 1 child, we're that awesome
    number_of_children = max(1, number_of_children)

    # add our chromosome to the sample
    sample.push_tail(json.dumps(chromosome))

    # our sample size has to be at least two large
    # (it takes two to tango)
    min_sample_size = 2

    print '[MC] sample_len: %s' % len(sample)

    # if the sample is large enough, pick two chromosome's to mate
    if len(sample) >= min_sample_size:

        # pop two chromosomes from the sample
        chrom1, chrom2 = sample.pop_head(), sample.pop_head()

        print '[MC] mating parents [%s] %s / [%s] %s' % (type(chrom1), chrom1, type(chrom2), chrom2)

        # reintrocude whatever chromosome we got if we didn't
        # get both
        if chrom1 is None or chrom2 is None:
            if chrom1:
                sample.push_head(chrom1)
            if chrom2:
                sample.push_head(chrom2)

        else:

            # de-serialize
            chrom1, chrom2 = map(json.loads, (chrom1, chrom2))

            print '[MC] number_of_children: %s' % number_of_children

            # split the chromosomes and mate them
            for i in xrange(number_of_children):
                child = _create_child(chrom1, chrom2)

                # yield up an event for each new chromosome
                yield dict( chromosome = child,
                            parents = [chrom1, chrom2],
                            run_id = run_id )


def mutate_chromosome(chromosome, run_id, event, _string):
    """
    mutates the given chromosome
    """

    # we want to mutate a % of the population
    mutation_chance = .5

    # but we only want to mutate if the population is low
    target_population_size = _int(_string('target_population_size:'+run_id))
    population_size = _int(_string('population_size:'+run_id))
    diff = target_population_size - population_size
    over_population_size = diff < 0

    if over_population_size:
        print '[MU] over_population_size: %s' % diff
    else:
        print '[MU] under_population_size: %s' % diff

    # should this chromosome mutate?
    if not over_population_size or r.random() < mutation_chance:

        # the number of genes that we mutate is based on the
        # mutation severity
        mutation_severity = 0.1 # % of genes mutated

        # get our image constraints
        image_size_x = _int(_string('image_size:x:'+run_id))
        image_size_y = _int(_string('image_size:y:'+run_id))
        image_size = [image_size_x, image_size_y]

        print '[MU] mutation_severity: %s' % mutation_severity

        # keep a copy for later
        original_chromosome = chromosome[:]
        for i in xrange(len(original_chromosome)):
            if r.random() <= mutation_severity:
                # flip the gene between 0 and 1
                chromosome[i] = int(not chromosome[i])

        yield dict( chromosome = chromosome,
                    original_chromosome = original_chromosome,
                    run_id = run_id )


def reintroduce_mutant(chromosome, run_id):
    yield True

def increment_population(_string, run_id):
    population = _string('population_size:'+run_id)
    p = population.incr()
    print '[I] population_size %s' % str(p)
    yield dict( population = p )

def decrement_population(_string, run_id):
    population = _string('population_size:'+run_id)
    p = population.decr()
    print '[D] population_size %s' % str(p)
    yield dict( population = p )

def event_counter(event_name, event_data, run_id, _dict):
    counts = _dict('event_counts:'+str(run_id))
    v = counts.incr(event_name)
    t = counts.incr('total')
    print '[EC] (%s) %s %s' % (t, event_name, v)
    yield False

print_count = 0
def print_revent_report(introspect, revent_client):
    global print_count
    if print_count % 100 == 0:
        introspect.print_report(revent_client)
    print_count += 1
    yield False


# NOTE: It feels like there is somethign wrong w/ this flow
app  = EventApp('ga_pic',

                # save to redis our run's params
                ('start_pic_ga', note_params, 'pic_ga_started'),

                # create our initial population
                (create_initial_population, 'created_chromosome'),

                # calculate the cost of all new chromosomes
                ('created_chromosome', calculate_cost, 'calculated_chromosome_cost'),

                # identify blow avg cost chromosomes
                (filter_costly_chromosomes, 'found_below_cost_chromosome'),

                # mate the good chromosomes, only good looking ppl get laid
                (mate_chromosomes, 'created_chromosome'),

                # mutate chromosomes, this will also mutate the initial population
                ('found_below_cost_chromosome', mutate_chromosome, 'mutated_chromosome'),

                # once we've mutated a chromosome we want to re-introduce it
                # to the pool as a created chromosome
                # NOTE: this works around each handler only being able to put
                #       off a single event type easily, I also bet revent
                #       would loop it back into the same queue, need to fix
                (reintroduce_mutant, 'created_chromosome'),

                # we want to keep a estimate of the total population
                ('created_chromosome', increment_population, 'incremented_population'),
                ('found_above_cost_chromosome', decrement_population, 'decremented_population'),

                ('.*', event_counter, '_'),
                ('.*', print_revent_report, '_')

)
app.run()
