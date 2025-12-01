import os
import polib

OLD = "PsychoPy"
NEW = "WrysysLab Studio"

def process_po_file(po_path):
    print(f"Processing: {po_path}")
    po = polib.pofile(po_path)

    changed = False

    for entry in po:
        # 只改译文，不改原文
        if entry.msgstr and OLD in entry.msgstr:
            entry.msgstr = entry.msgstr.replace(OLD, NEW)
            changed = True

        # 如果有复数形式，也一并处理
        if entry.msgstr_plural:
            for idx, text in entry.msgstr_plural.items():
                if OLD in text:
                    entry.msgstr_plural[idx] = text.replace(OLD, NEW)
                    changed = True

    if changed:
        po.save(po_path)
        print(f"  Updated: {po_path}")
    else:
        print(f"  No changes: {po_path}")

    # 编译为同名 .mo
    mo_path = os.path.splitext(po_path)[0] + ".mo"
    po.save_as_mofile(mo_path)
    print(f"  Compiled to: {mo_path}")


def main():
    current_dir = os.getcwd()
    print("Current directory:", current_dir)

    po_files = [f for f in os.listdir(current_dir) if f.lower().endswith(".po")]

    if not po_files:
        print("No .po files found in current directory.")
        return

    for po_name in po_files:
        po_path = os.path.join(current_dir, po_name)
        process_po_file(po_path)

    print("All done.")


if __name__ == "__main__":
    main()
