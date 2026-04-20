import glob
import sys
from pathlib import Path

def resolve_image_paths(args):
    """
    Risolve ricorsivamente le occorrenze dei filepath in base agli argomenti.
    Supporta wildcard (*, ?) e percorsi relativi in vari formati.
    """
    image_paths = []
    for arg in args:
        if '*' in arg or '?' in arg:
            matches = sorted(glob.glob(arg))
            image_paths.extend(matches)
            
            if not matches:
                alt_arg = f"dataset/{arg}"
                matches = sorted(glob.glob(alt_arg))
                image_paths.extend(matches)
            
            if not matches:
                alt_arg2 = arg.replace("archive/", "")
                matches = sorted(glob.glob(alt_arg2))
                image_paths.extend(matches)
            
            if not matches:
                alt_arg3 = f"dataset/{arg.replace('archive/', '')}"
                matches = sorted(glob.glob(alt_arg3))
                image_paths.extend(matches)
            
            if not matches:
                alt_arg4 = f"../{arg}"
                matches = sorted(glob.glob(alt_arg4))
                image_paths.extend(matches)
        else:
            if Path(arg).exists():
                image_paths.append(arg)
            else:
                alt_path = Path(f"dataset/{arg}")
                if alt_path.exists():
                    image_paths.append(str(alt_path))
                else:
                    alt_path2 = Path(arg.replace("archive/", ""))
                    if alt_path2.exists():
                        image_paths.append(str(alt_path2))
                    else:
                        alt_path3 = Path(f"dataset/{arg.replace('archive/', '')}")
                        if alt_path3.exists():
                            image_paths.append(str(alt_path3))
                        else:
                            alt_path4 = Path(f"../{arg}")
                            if alt_path4.exists():
                                image_paths.append(str(alt_path4))
    
    if not image_paths and len(args) > 0:
        arg = args[0]
        matches = sorted(glob.glob(arg))
        if matches:
            image_paths = matches
        else:
            alt_arg = f"dataset/{arg}"
            matches = sorted(glob.glob(alt_arg))
            if matches:
                image_paths = matches
            else:
                alt_arg2 = arg.replace("archive/", "")
                matches = sorted(glob.glob(alt_arg2))
                if matches:
                    image_paths = matches
                else:
                    alt_arg3 = f"dataset/{arg.replace('archive/', '')}"
                    matches = sorted(glob.glob(alt_arg3))
                    if matches:
                        image_paths = matches
                    else:
                        alt_arg4 = f"../{arg}"
                        matches = sorted(glob.glob(alt_arg4))
                        if matches:
                            image_paths = matches
                            
    return image_paths