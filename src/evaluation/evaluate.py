import argparse
from evaluate_celebrity import extract_answers_prompt_celebrity, detect_retraction_prompt_celebrity, evaluate_free_celebrity, evaluate_continuation_celebrity
from evaluate_wikidata import extract_answers_prompt_wikidata, detect_retraction_prompt_wikidata, evaluate_free_wikidata, evaluate_continuation_wikidata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', type=str)
    parser.add_argument('--name_filename', type=str)
    parser.add_argument('--retraction_filename', type=str)
    parser.add_argument('--continuation', action='store_true')
    parser.add_argument('--func', type=str, default='extract_answers_prompt')
    parser.add_argument('--data_source', type=str, default='wikidata')
    args = parser.parse_args()

    assert args.data_source in args.input_filename

    if args.func == 'extract_answers_prompt':
        if args.data_source == 'wikidata':
            extract_answers_prompt_wikidata(args.input_filename)
        elif args.data_source == 'celebrity':
            extract_answers_prompt_celebrity(args.input_filename)
        else:
            raise NotImplementedError
    elif args.func == 'detect_retraction_prompt':
        if args.data_source == 'wikidata':
            detect_retraction_prompt_wikidata(args.input_filename, args.name_filename, continuation=args.continuation)
        elif args.data_source == 'celebrity':
            detect_retraction_prompt_celebrity(args.input_filename, args.name_filename, continuation=args.continuation)
        else:
            raise NotImplementedError
    elif args.func == 'evaluate_free':
        if args.data_source == 'wikidata':
            evaluate_free_wikidata(args.input_filename, args.name_filename, args.retraction_filename)
        elif args.data_source == 'celebrity':
            evaluate_free_celebrity(args.input_filename, args.name_filename, args.retraction_filename)
        else:
            raise NotImplementedError
    elif args.func == 'evaluate_continuation':
        if args.data_source == 'wikidata':
            evaluate_continuation_wikidata(args.input_filename, args.retraction_filename)
        elif args.data_source == 'celebrity':
            evaluate_continuation_celebrity(args.input_filename, args.retraction_filename)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
