import os
import collections
import importlib
import pickle
import pdb

def import_class(_class):
    if type(_class) is not str: return _class
    ## 'diffusion' on standard installs
    repo_name = __name__.split('.')[0]
    ## eg, 'utils'
    module_name = '.'.join(_class.split('.')[:-1])
    ## eg, 'Renderer'
    class_name = _class.split('.')[-1]
    ## eg, 'diffusion.utils'
    module = importlib.import_module(f'{repo_name}.{module_name}')
    ## eg, diffusion.utils.Renderer
    _class = getattr(module, class_name)
    print(f'[ utils/config ] Imported {repo_name}.{module_name}:{class_name}')
    return _class

class Config(collections.abc.Mapping):

    def __init__(self, _class, verbose=True, savepath=None, device=None, **kwargs):
        self._class = import_class(_class)
        self._device = device
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
            self.savepath = savepath
            pickle.dump(self, open(savepath, 'wb'))
            print(f'[ utils/config ] Saved config to: {savepath}\n')

    def __repr__(self):
        string = f'\n[utils/config ] Config: {self._class}\n'
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            if key == 'problems_dict': # (2, 10, 20, 6), too long, print a small part
                if val is not None:
                    val = {k: v[0, :2] if v.ndim >= 2 else v[:2] for k, v in val.items()}
            string += f'    {key}: {val}\n'
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, attr):
        if attr == '_dict' and '_dict' not in vars(self):
            self._dict = {}
            return self._dict
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        instance = self._class(*args, **kwargs, **self._dict)
        if self._device:
            instance = instance.to(self._device)
        # pdb.set_trace()
        if getattr(self, 'savepath', None):
            if 'model_config.pkl' in self.savepath:
                m_path = self.savepath.replace('model_config.pkl', 'model_config.txt')
                with open(m_path, 'w') as f:
                    print(instance, file=f)
        return instance
