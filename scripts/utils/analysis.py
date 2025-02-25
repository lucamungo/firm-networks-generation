def print_comparison_table(name, original, generated, difference=None):
    """Print a nicely formatted comparison table."""
    print(f"\n{name}:")
    print("-" * 70)
    print(f"{'Property':<30} {'Original':>12} {'Generated':>12} {'Diff':>12}")
    print("-" * 70)

    for k in original.keys():
        orig_val = original[k]
        gen_val = generated[k]
        diff = gen_val - orig_val if difference is not None else None

        if difference is not None:
            print(f"{k:<30} {orig_val:>12.3f} {gen_val:>12.3f} {diff:>12.3f}")
        else:
            print(f"{k:<30} {orig_val:>12.3f} {gen_val:>12.3f}")
    print("-" * 70)


def print_loss_comparison(initial_losses, final_losses):
    """Print a comparison of losses before and after training."""
    print("\nLoss Comparison:")
    print("-" * 102)
    print(f"{'Component':<30} {'Initial':>12} {'Final':>12} {'Abs Change':>12} {'% Change':>12}")
    print("-" * 102)

    # First print main loss components
    main_components = ["correlation", "hill", "io", "smooth"]
    for k in main_components:
        if k in initial_losses and k in final_losses:
            init_val = initial_losses[k].item()
            final_val = final_losses[k].item()
            abs_change = final_val - init_val
            pct_change = (abs_change / init_val) * 100 if init_val != 0 else float("inf")

            print(
                f"{k:<30} {init_val:>12.4f} {final_val:>12.4f} {abs_change:>12.4f} {pct_change:>11.1f}%"
            )

    print("-" * 102)

    # Then print detailed components, grouped by type
    groups = {
        "Correlation Components": "correlation_",
        "Hill Components": "hill_",
        "Smoothness Components": "smooth_",
    }

    for group_name, prefix in groups.items():
        components = [k for k in initial_losses.keys() if k.startswith(prefix)]
        if components:
            print(f"\n{group_name}:")
            print("-" * 102)
            for k in components:
                if k in final_losses:
                    init_val = initial_losses[k].item()
                    final_val = final_losses[k].item()
                    abs_change = final_val - init_val
                    pct_change = (abs_change / init_val) * 100 if init_val != 0 else float("inf")

                    # Remove the prefix for cleaner display
                    display_name = k[len(prefix) :]
                    print(
                        f"{display_name:<30} {init_val:>12.4f} {final_val:>12.4f} {abs_change:>12.4f} {pct_change:>11.1f}%"
                    )
            print("-" * 102)
