"""
Reads in catalogs, typically for the other script (graphspectrum) to use.
"""

import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
import pickle

def _helper_pair(args):
    '''
    A quick helper function to create unique IDs inside of catalogs
    '''
    
    cantor = lambda a,b : 0.5*(a+b)*(a+b+1)+b
    
    if len(args)>2:
        res = cantor(_helper_pair(args[:-1]), args[-1])
    else:
        res = cantor(*args)
    
    
    return res.astype(int)

def read_at20g(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['NAME']
    
    # Coordinate fixes
    new['RA'] = dat['RA']
    new['RA_Err'] = np.full(length, 0.9/3600) * u.deg
    new['Dec'] = dat['DEC']
    new['Dec_Err'] = np.full(length, 1.0/3600) * u.deg
    
    # First set
    new['Freq'] = dat['freq']
    new['Peak_Flux'] = dat['peak_flux']
    new['Peak_Flux_Err'] = dat['peak_flux_error']
    
    new['Int_Flux'] = dat['peak_flux'].data * u.Unit('mJy')
    new['Int_Flux_Err'] = dat['peak_flux_error'].data * u.Unit('mJy')
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_cgps(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['NAME']
    
    # Coordinate fixes
    new['RA'] = dat['RA'].data * u.deg
    new['RA_Err'] = dat['RA_ERROR'].data * (15/3600) * u.deg # sec -> arcsec
    new['Dec'] = dat['DEC'].data * u.deg
    new['Dec_Err'] = dat['DEC_ERROR'].data * (1/3600) * u.deg
    
    # First set
    new['Freq'] = np.full(length, 1.42e9) * u.Hz
    new['Peak_Flux'] = dat['FLUX_1420_MHZ'].data * u.Unit('mJy/beam')
    new['Peak_Flux_Err'] = dat['FLUX_1420_MHZ_ERROR'].data * u.Unit('mJy/beam')
    
    new['Int_Flux'] = dat['INT_FLUX_1420_MHZ'].data * u.Unit('mJy')
    new['Int_Flux_Err'] = dat['INT_FLUX_1420_MHZ_ERROR'].data * u.Unit('mJy')
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_first(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['FIRST']
    
    # For uncertianty
    dat['Maj'][dat['Maj']==0] = 8.117 # mean + 1 std, conversative estimate
    
    # Coordinate fixes
    unc = dat['Maj'] * (dat['Rms']/(dat['Fpeak']-0.25) + 1/20)
    new['RA'] = dat['RAJ2000']
    new['RA_Err'] = unc.to('deg') 
    new['Dec'] = dat['DEJ2000']
    new['Dec_Err'] = unc.to('deg')
    
    # First set
    new['Freq'] = np.full(length, 1.4e9) * u.Hz
    new['Peak_Flux'] = dat['Fpeak'].data * u.Unit('mJy/beam')
    new['Peak_Flux_Err'] = dat['Rms'].data * u.Unit('mJy/beam')
    
    new['Int_Flux'] = dat['Fint']
    beams = new['Int_Flux']/new['Peak_Flux']
    new['Int_Flux_Err'] = (new['Peak_Flux_Err'] * beams).data * u.mJy
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_frnk(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Generate Unique ID
    id1 = dat['Isl_id']
    id2 = np.unique(dat['project'], return_inverse=True)[1]
    id3 = np.unique(dat['block'], return_inverse=True)[1]
    id4 = np.unique(dat['field'], return_inverse=True)[1]
    large_usid = _helper_pair([id1,id2,id3,id4])
    usid = np.unique(large_usid, return_inverse=True)[1]
    new['USID'] = usid
    new['Name'] = [ f'{i:04d}' for i in np.arange(0,len(dat)) ]
    
    # Coordinates
    new['RA'] = dat['RA']
    new['RA_Err'] = dat['E_RA']
    new['Dec'] = dat['DEC']
    new['Dec_Err'] = dat['E_DEC']
    
    # Define a safe beam correction
    bc = dat['BeamCorrection']
    bc[dat['IsOutsideBeam']] = 1
    
    # Freqeuncy # FIX ME # # BEAM CORRECT ME #
    new['Freq'] = dat['freq']
    new['Peak_Flux'] = dat['Peak_flux'].to('mJy/beam') / bc
    new['Peak_Flux_Err'] = dat['E_Peak_flux'].to('mJy/beam') / bc
    new['Int_Flux'] = dat['Total_flux'].to('mJy') / bc
    new['Int_Flux_Err'] = dat['E_Total_flux'].to('mJy') / bc
    
    new['S_Code'] = dat['S_Code']
    
    # Lower limit column
    new['IsLowerLim'] = dat['IsOutsideBeam']
    
    return new

def read_gleam(filename):
    
    dat_ext = Table.read(filename, hdu=1)
    dat_gal = Table.read(filename, hdu=2)
    dat = vstack([dat_ext, dat_gal], metadata_conflicts='silent')
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['GLEAM']
    
    # Coordinate fixes
    new['RA'] = dat['RAJ2000']
    new['RA_Err'] = dat['e_RAJ2000']
    new['Dec'] = dat['DEJ2000']
    new['Dec_Err'] = dat['e_DEJ2000']
    
    # First set
    new['Freq'] = np.full(length, 200e6) * u.Hz # FIX
    new['Peak_Flux'] = dat['Fpwide'].to('mJy/beam')
    new['Peak_Flux_Err'] = dat['e_Fpwide'].to('mJy/beam')
    new['Int_Flux'] = dat['Fintwide'].to('mJy')
    new['Int_Flux_Err'] = dat['e_Fintwide'].to('mJy')
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_lotss(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['Source']
    
    # Coordinate fixes
    new['RA'] = dat['RAJ2000']
    new['RA_Err'] = dat['e_RAJ2000'].to('deg')
    new['Dec'] = dat['DEJ2000']
    new['Dec_Err'] = dat['e_DEJ2000'].to('deg')
    
    # First set
    new['Freq'] = np.full(length, 144e6) * u.Hz # FIX
    new['Peak_Flux'] = dat['Speak'].to('mJy/beam')
    new['Peak_Flux_Err'] = dat['e_Speak'].to('mJy/beam')
    new['Int_Flux'] = dat['Sint'].to('mJy')
    new['Int_Flux_Err'] = dat['e_Sint'].to('mJy')
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_nvss(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['NVSS']
    
    # Coordinate fixes
    new['RA'] = dat['_RAJ2000']
    new['RA_Err'] = dat['e_RAJ2000'].data*15/3600 * u.deg
    new['Dec'] = dat['_DEJ2000']
    new['Dec_Err'] = dat['e_DEJ2000'].to('deg')
    
    # First set
    new['Freq'] = np.full(length, 1.4e9) * u.Hz # FIX
    new['Int_Flux'] = dat['S1_4'].to('mJy')
    new['Int_Flux_Err'] = dat['e_S1_4'].to('mJy')
    
    new['Peak_Flux'] = new['Int_Flux'].data * u.Unit('mJy/beam')
    new['Peak_Flux_Err'] = new['Int_Flux_Err'].data * u.Unit('mJy/beam')
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_pmn(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['NAME']
    
    # Coordinate fixes
    new['RA'] = dat['RA'].data * u.deg
    new['RA_Err'] = np.full(length, 1.5/3600) * u.deg
    new['Dec'] = dat['DEC'].data * u.deg
    new['Dec_Err'] = np.full(length, 1/3600) * u.deg
    
    # First set
    new['Freq'] = np.full(length, 4.85e9) * u.Hz # FIX
    new['Int_Flux'] = dat['FLUX_4850_MHZ'].data * u.mJy
    new['Int_Flux_Err'] = dat['FLUX_4850_MHZ_ERROR'].data * u.mJy
    
    new['Peak_Flux'] = new['Int_Flux'].data * u.Unit('mJy/beam')
    new['Peak_Flux_Err'] = new['Int_Flux_Err'].data * u.Unit('mJy/beam')
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new
    
def read_sumss(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['NAME']
    
    # Coordinate fixes
    new['RA'] = dat['RA'].data * u.deg
    new['RA_Err'] = dat['RA_ERROR'].data/3600 * u.deg
    new['Dec'] = dat['DEC'].data * u.deg
    new['Dec_Err'] = dat['DEC_ERROR'].data/3600 * u.deg
    
    # First set
    new['Freq'] = np.full(length, 8.43e8) * u.Hz # FIX
    new['Peak_Flux'] = dat['FLUX_36_CM'].data * u.Unit('mJy/beam')
    new['Peak_Flux_Err'] = dat['FLUX_36_CM_ERROR'].data * u.Unit('mJy/beam')
    new['Int_Flux'] = dat['INT_FLUX_36_CM'].data * u.mJy
    new['Int_Flux_Err'] = dat['INT_FLUX_36_CM_ERROR'].data * u.mJy
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_tgss(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['Source_name']
    
    # Coordinate fixes
    new['RA'] = dat['RA']
    new['RA_Err'] = dat['E_RA'].to('deg')
    new['Dec'] = dat['DEC']
    new['Dec_Err'] = dat['E_DEC'].to('deg')
    
    # First set
    new['Freq'] = np.full(length, 1.50e8) * u.Hz # FIX
    new['Peak_Flux'] = dat['Peak_flux']
    new['Peak_Flux_Err'] = dat['E_Peak_flux']
    new['Int_Flux'] = dat['Total_flux']
    new['Int_Flux_Err'] = dat['E_Total_flux']
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_vlass(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['Name']
    
    # Generate the unique source idea
    id1 = dat['Isl_id']
    id2 = np.unique(dat['Pointing'], return_inverse=True)[1]
    large_usid = _helper_pair([id1,id2])
    usid = np.unique(large_usid, return_inverse=True)[1]
    new['USID'] = usid
    
    # Coordinate fixes
    new['RA'] = dat['RA']
    new['RA_Err'] = dat['E_RA'].to('deg')
    new['Dec'] = dat['DEC']
    new['Dec_Err'] = dat['E_DEC'].to('deg')
    
    # First set
    new['Freq'] = np.full(length, 3e9) * u.Hz # FIX
    new['Peak_Flux'] = dat['Peak_flux'].to('mJy/beam')
    new['Peak_Flux_Err'] = dat['E_Peak_flux'].to('mJy/beam')
    new['Int_Flux'] = dat['Total_flux'].to('mJy')
    new['Int_Flux_Err'] = dat['E_Total_flux'].to('mJy')
    
    # Get S_Code Column
    new['S_Code'] = dat['S_Code']
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    # Get rid of that one source with a NaN position error
    valued = np.where(~np.isnan(new['RA_Err']))[0]
    new = new[valued]
    
    return new

def read_vlite(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = [ str(n) for n in dat['VLITE_id']]
    
    # Coordinate fixes
    new['RA'] = dat['RA']
    new['RA_Err'] = dat['E_RA']
    new['Dec'] = dat['Dec']
    new['Dec_Err'] = dat['E_Dec']
    
    # First set
    new['Freq'] = np.full(length, 3.64e8) * u.Hz # FIX
    new['Peak_Flux'] = dat['PeakFlux']
    new['Peak_Flux_Err'] = dat['E_PeakFlux']
    new['Int_Flux'] = dat['TotalFlux']
    new['Int_Flux_Err'] = dat['E_TotalFlux']
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new

def read_vlssr(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['NAME']
    
    # Coordinate fixes
    new['RA'] = dat['RA'].data * u.deg
    new['RA_Err'] = np.full(length, 3/3600) * u.deg
    new['Dec'] = dat['DEC'].data * u.deg
    new['Dec_Err'] = np.full(length, 3.4/3600) * u.deg
    
    # First set
    new['Freq'] = np.full(length, 7.4e7) * u.Hz # FIX
    new['Peak_Flux'] = dat['FLUX_74_MHZ'].data * u.Unit('mJy/beam')
    new['Peak_Flux_Err'] = dat['FLUX_74_MHZ_ERROR'].data * u.Unit('mJy/beam')
    
    new['Int_Flux'] = new['Peak_Flux'].data * u.mJy
    new['Int_Flux_Err'] = new['Peak_Flux_Err'].data * u.mJy
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new
    
def read_wenss(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['NAME']
    
    # Coordinate fixes
    new['RA'] = dat['RA'].data * u.deg
    new['RA_Err'] = np.full(length, 0.15/3600) * u.deg
    new['Dec'] = dat['DEC'].data * u.deg
    new['Dec_Err'] = np.full(length, 0.1/3600) * u.deg
    
    # First set
    new['Freq'] = np.full(length, 3.25e8) * u.Hz # FIX
    new['Peak_Flux'] = dat['FLUX_92_CM'].data * u.Unit('mJy/beam')
    new['Peak_Flux_Err'] = dat['FLUX_92_CM_ERROR'].data * u.Unit('mJy/beam')
    new['Int_Flux'] = dat['INT_FLUX_92_CM'].data * u.mJy
    beams = new['Int_Flux']/new['Peak_Flux']
    new['Int_Flux_Err'] = (dat['FLUX_92_CM_ERROR'].data * beams).data * u.mJy
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new 

def read_wish(filename):
    
    dat = Table.read(filename)
    new = Table()
    
    # Helper stuff
    length = len(dat)
    
    # Get the names
    new['Name'] = dat['NAME']
    
    # Coordinate fixes
    new['RA'] = dat['RA'].data * u.deg
    new['RA_Err'] = np.full(length, 0.15/3600) * u.deg
    new['Dec'] = dat['DEC'].data * u.deg
    new['Dec_Err'] = np.full(length, 0.1/3600) * u.deg
    
    # First set
    new['Freq'] = np.full(length, 3.52e8) * u.Hz # FIX
    new['Peak_Flux'] = dat['FLUX_352_MHZ'].data * u.Unit('mJy/beam')
    new['Peak_Flux_Err'] = dat['FLUX_352_MHZ_ERROR'].data * u.Unit('mJy/beam')
    
    new['Int_Flux'] = dat['INT_FLUX_352_MHZ'].data * u.mJy
    beams = new['Int_Flux']/new['Peak_Flux']
    new['Int_Flux_Err'] = (dat['FLUX_352_MHZ_ERROR'].data * beams).data * u.mJy
    
    # Lower limit column
    new['IsLowerLim'] = np.full(len(dat), False)
    
    return new 

def reader(filename):
    '''
    Reads in an input fits file and cleans it up as best it can. Unused
    columns are removed, used columns are renamed, and units are all coverted 
    more useful standards.
    
    ----Inputs----
    filename (str) = the filepath to the data table to be read in. Expects 
        data to be in fits format, and to have been renamed using my
        SURVEY.fits naming convention
    
    ----Outputs----
    trim_dat (Table) = an Astropy Table containing the reformatted and cleaned
        up data.
    '''
    
    file = filename.split('/')[-1]
    cat_name = file[:file.find('.')].lower()
    
    func_map = {'at20g_05': read_at20g, 'at20g_08': read_at20g, 
                'at20g_20': read_at20g, 'cgps': read_cgps, 'first': read_first, 
                'd5ghz': read_frnk, 'd7ghz': read_frnk,  'gleam': read_gleam, 
                'lotss': read_lotss, 'nvss': read_nvss, 'pmn': read_pmn, 
                'sumss': read_sumss, 'tgss': read_tgss, 'vlass': read_vlass,
                'vlite': read_vlite, 'vlssr': read_vlssr, 'wenss': read_wenss, 
                'wish': read_wish}
    
    cols = ['Name', 'RA', 'RA_Err', 'Dec', 'Dec_Err', 'Freq', 'Peak_Flux', 
            'Peak_Flux_Err', 'Int_Flux', 'Int_Flux_Err']
    
    if cat_name in func_map:
        nice_cat = func_map[cat_name](filename)
        
        # Verify columns 
        for col in cols:
            if not col in nice_cat.colnames:
                nice_cat[col] = np.full(len(nice_cat), np.nan)
                
            if not 'USID' in nice_cat.colnames:
                nice_cat['USID'] = np.arange(0,len(nice_cat))
                
        # Some of sources stupidly have 0 position error
        bad_vals = np.full(len(nice_cat), False)
        nonzero_cols = ['RA_Err', 'Dec_Err', 'Peak_Flux', 'Peak_Flux_Err', 
                        'Int_Flux', 'Int_Flux_Err']
        for col in nonzero_cols:
            bad_vals |= nice_cat[col]==0
        nice_cat = nice_cat[~bad_vals]
        
        # Add survey name columns
        nice_cat['survey'] = np.full(len(nice_cat), cat_name)
        
    else:
        raise ValueError('I don\'t know about that catalog: '+cat_name)
    
    return nice_cat

def skyplots(files, dims=(4,4), size=(12,7)):
    
    import matplotlib.pyplot as plt
    
    maxlen = 1e5
    
    x, y = dims
    plt.figure(figsize=size, dpi=100)
    for i in range(len(files)):
        file = files[i]
        if file.endswith('AT20G_20.fits'):
            name = 'AT20G'
        elif file.endswith('D7GHZ.fits'):
            name = 'Dedicated'
        else:
            name = file.split('/')[-1].replace('.fits','')
            
        dat = reader(file)
        
        if len(dat)>maxlen:
            subset = np.random.choice(np.arange(0,len(dat)), int(maxlen), 
                                      replace=False)
            dat = dat[subset]
            
        sc = SkyCoord(dat['RA'], dat['Dec'])
        plt.subplot2grid((y,x),(i//x, i%x), projection='mollweide')
        plt.scatter(sc.ra.wrap_at('180d').rad, sc.dec.rad, s=1, c=f'C{i}')
        plt.title(name)
        plt.grid()
        plt.xticks(np.linspace(-np.pi,np.pi,7),[])
        plt.yticks(np.linspace(-np.pi/2,np.pi/2,7),[])
    plt.tight_layout()
    plt.savefig('prettyskies.png')
    plt.show()
        
        
        

if __name__=='__main__':
    
    catdir = os.getcwd() + '/catalogs/radio/'
    
    import matplotlib.pyplot as plt
    shorthand = ['D5GHZ.fits', 'D7GHZ.fits', 'VLASS.fits']
    catalogs = {}
    
    r = 1 # DEG
    lbins = np.linspace(-np.pi, np.pi, 360//r)
    bbins = np.arcsin(np.linspace(-1,1, 180//r))
    
    if os.path.exists('themap.p'):
        themap = pickle.load(open('themap.p', 'rb'))
    else:
        themap = np.zeros((lbins.size-1, bbins.size-1))
        for file in sorted(os.listdir(catdir)):
            print('='*15, file, '='*15)
            
            if '.txt' in file:
                continue
            if 'FERMI' in file:
                continue
            stuff = reader(catdir+file)
            catalogs[file.replace('.fits','')] = stuff
            
            sc = SkyCoord(stuff['RA'], stuff['Dec'])
            l = sc.galactic.l.wrap_at('180d').rad
            b = sc.galactic.b.rad
            
            C,_,_ = np.histogram2d(l, b, bins=(lbins, bbins))
            C[C>0] = 1
            themap += C
            # break
        pickle.dump(themap, open('themap.p', 'wb'))
        
    
    fig = plt.figure(figsize=(8,5), dpi=240)
    ax = fig.add_subplot(projection='hammer')
    plt.pcolormesh(-lbins,bbins,themap.T, zorder=3, vmin=0, vmax=9)
    plt.colorbar(orientation='horizontal', extend='max', shrink=0.9, pad=0.05, 
                 label='Number of Catalogs Covering')
    # North Hemisphere circle
    circle = SkyCoord(np.linspace(0, np.pi*2), np.full(50, np.deg2rad(-40)), unit='rad,rad')
    plt.plot(-circle.galactic.l.wrap_at('180d').rad, circle.galactic.b.rad, 'w--', 
              lw=2, zorder=5)
    
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('/Users/bruzewskis/Dropbox/coveragemap.png', bbox_inches='tight')
    plt.show()
    
    # fig = plt.figure(figsize=(8,9), dpi=3840/9)
    # for i in range(6):
    #     ax = plt.subplot2grid((3,2), (i//2, i%2), projection='hammer', fig=fig)
    #     plt.pcolormesh(-lbins, bbins, themap.T, vmin=0, vmax=12-i)
    #     plt.title(f'vmax={12-i}')
    #     plt.colorbar(orientation='horizontal', extend='max')
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('/Users/bruzewskis/Dropbox/testingcontrast.png', bbox_inches='tight')
        
