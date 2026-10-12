import sys
import vinfo
from datetime import date

version = vinfo.version
release = vinfo.release
patch = vinfo.patch

def get_vstr(version, release, patch):
    vstr = str(version) + '.'
    vstr += str(release) + '.'
    vstr += str(patch)
    return vstr

# run using: python vcon mode
# where mode = patch, release or version depending on the version update you
# require.

if __name__ == "__main__":

    print("")
    print("Running Version Controller")
    print("--------------------------")

    vstr_before = get_vstr(version, release, patch)

    mode = str(sys.argv[1])

    if mode == 'patch':
        patch += 1
    elif mode == 'release':
        patch = 0
        release += 1
    elif mode == 'version':
        patch = 0
        release = 0
        version += 1
    elif mode == 'test':
        pass

    print("")
    print("Mode = ", mode)

    vstr_after = get_vstr(version, release, patch)

    print("")
    print("Current version:", vstr_before)
    print("Updated version:", vstr_after)

    with open('vinfo.py', 'w') as file:
        file.write('version='+str(version)+'\n')
        file.write('release='+str(release)+'\n')
        file.write('patch='+str(patch)+'\n')
        file.write('vstr="'+vstr_after+'"\n')
    
    print("")
    print("Update CITATION.cff")

    release_date = date.today().isoformat()

    with open("CITATION.cff", "r", encoding="utf-8") as file:
        lines = file.readlines()

    has_release_date = False

    for i, line in enumerate(lines):

        if line.startswith("version:"):
            lines[i] = f'version: "{vstr_after}"\n'

        elif line.startswith("date-released:"):
            has_release_date = True
            if mode != "test":
                lines[i] = f'date-released: "{release_date}"\n'

    # Add the release date if the field doesn't already exist.
    if not has_release_date and mode != "test":
        lines.append(f'date-released: "{release_date}"\n')

    with open("CITATION.cff", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("")
    print("Update VERSION")

    setup_file = open("VERSION", "r")
    list_of_lines = setup_file.readlines()
    for i in range(0, 1):
        list_of_lines[i] = '%s' % vstr_after
    setup_file.close()

    setup_file = open("VERSION", "w")
    setup_file.writelines(list_of_lines)
    setup_file.close()

    print("")
