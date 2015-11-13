# Author::    Lucas Carlson  (mailto:lucas@rufy.com)
# Copyright:: Copyright (c) 2005 Lucas Carlson
# License::   LGPL

require 'fast_stemmer'
require 'stopwords'

module ClassifierReborn
  class Bayes
    CategoryNotFoundError = Class.new(StandardError)

    attr_accessor :threshold

    # The class can be created with one or more categories, each of which will be
    # initialized and given a training method. E.g.,
    #      b = ClassifierReborn::Bayes.new 'Interesting', 'Uninteresting', 'Spam'
    #
    # Options available are:
    #   language:         'en'   Used to select language specific stop words
    #   auto_categorize:  false  When true, enables ability to dynamically declare a category
    #   enable_threshold: false  When true, enables a threshold requirement for classifition
    #   threshold:        0.0    Default threshold, only used when enabled
    def initialize *args, language: 'en', auto_categorize: false, enable_threshold: false, threshold: 0.0
      @categories = {}

      args.flatten.each do |arg|
        add_category arg
      end

      @total_words         = 0
      @category_counts     = {}
      @category_word_count = {}

      @language            = language
      @auto_categorize     = auto_categorize
      @enable_threshold    = enable_threshold
      @threshold           = threshold
    end

    # Provides a general training method for all categories specified in Bayes#new
    # For example:
    #     b = ClassifierReborn::Bayes.new 'This', 'That', 'the_other'
    #     b.train :this, "This text"
    #     b.train "that", "That text"
    #     b.train "The other", "The other text"
    def train(category, text)
      # Add the category dynamically or raise an error
      add_category(category) if @auto_categorize

      unless @categories.has_key?(category)
        fail CategoryNotFoundError, "Cannot train; category #{ category } does not exist"
      end

      @category_counts[category] += 1
      word_hash(text).each do |word, count|
        @categories[category][word] += count
        @category_word_count[category] += count
        @total_words += count
      end
    end

    # Provides a untraining method for all categories specified in Bayes#new
    # Be very careful with this method.
    #
    # For example:
    #     b = ClassifierReborn::Bayes.new 'This', 'That', 'the_other'
    #     b.train :this, "This text"
    #     b.untrain :this, "This text"
    def untrain(category, text)
      @category_counts[category] -= 1

      word_hash(text).each do |word, count|
        next if @total_words < 0

        orig = @categories[category][word] || 0
        @categories[category][word] -= count

        if @categories[category][word] <= 0
          @categories[category].delete(word)
          count = orig
        end

        @category_word_count[category] -= count if @category_word_count[category] >= count
        @total_words -= count
      end
    end

    # Returns the scores in each category the provided +text+. E.g.,
    #    b.classifications "I hate bad words and you"
    #    =>  {"Uninteresting"=>-12.6997928013932, "Interesting"=>-18.4206807439524}
    # The largest of these scores (the one closest to 0) is the one picked out by #classify
    def classifications(text)
      # TODO:XXX:TODO (ashby) : This can be something like an inject
      score = {}

      word_hash_cache = word_hash(text)
      training_count = @category_counts.values.reduce(:+).to_f

      @categories.each do |category, category_words|
        score[category.to_s] = 0
        total = (@category_word_count[category] || 1).to_f

        word_hash_cache.each do |word, _count|
          s = category_words.key?(word) ? category_words[word] : 0.1
          score[category.to_s] += Math.log(s / total)
        end

        # now add prior probability for the category
        s = @category_counts.key?(category) ? @category_counts[category] : 0.1
        score[category.to_s] += Math.log(s / training_count)
      end

      score
    end

    # Returns the classification of the provided +text+, which is one of the
    # categories given in the initializer along with the score. E.g.,
    #    b.classify "I hate bad words and you"
    #    =>  ['Uninteresting', -4.852030263919617]
    def classify_with_score(text)
      classifications(text).sort_by{ |a| -a[1] }.first
    end

    # Return the classification without the score
    def classify(text)
      result, score = classify_with_score(text)
      result = nil if score < @threshold || score == Float::INFINITY if threshold_enabled?
      result
    end

    # Dynamically enable threshold for classify results
    def enable_threshold!
      @enable_threshold = true
    end

    # Dynamically disable threshold for classify results
    def disable_threshold!
      @enable_threshold = false
    end

    # Is threshold processing enabled?
    def threshold_enabled?
      @enable_threshold
    end

    # is threshold processing disabled?
    def threshold_disabled?
      !@enable_threshold
    end

    # Provides a list of category names
    # For example:
    #     b.categories
    #     =>   ['This', 'That', 'the_other']
    def categories # :nodoc:
      @categories.keys
    end

    # Allows you to add categories to the classifier.
    # For example:
    #     b.add_category "Not spam"
    #
    # WARNING: Adding categories to a trained classifier will
    # result in an undertrained category that will tend to match
    # more criteria than the trained selective categories. In short,
    # try to initialize your categories at initialization.
    def add_category(category)
      @categories[category] ||= {}
    end

    private

    def stopword_filter
      @filter ||= Stopwords::Snowball::Filter.new @language
    end

    # Return a Hash of strings => ints. Each value is the stemmed version of a
    # word, with the value being its frequency in the document so long as it
    # isn't punctuation, a stopword or a very short word.
    def word_hash str
      str.gsub(/[^\p{WORD}\s]/, '').downcase.split.inject({}) do |memo, word|
        next memo unless word.length > 2
        next memo if stopword_filter.stopword? word

        memo[word.stem] += 1

        memo
      end
    end
  end
end
